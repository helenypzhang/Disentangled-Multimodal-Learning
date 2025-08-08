import os
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
from data.dataset import *
from torch.utils.data import DataLoader
from train_test import testDeformPathomicModel, testBaselineModel, testMultiScaleModel, testDistillation, testTeachersModel, testStudentsModel
from utils.yaml_config_hook import yaml_config_hook
from utils.sync_batchnorm import convert_model
from models.model import define_net
from sklearn.model_selection import StratifiedKFold, KFold


def main(gpu, args, wandb_logger):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # test_dataset_IvYGAP = IvYGAP_Dataset(phase='Test', args=args)
    # test_dataset_TCGA = TCGA_Dataset(phase='Test', args=args)
    # test_dataset = torch.utils.data.ConcatDataset([test_dataset_IvYGAP, test_dataset_TCGA])
    # for CPTAC:
    if args.external_eval:
        print("Now Inferencing CPTAC!")
        labels_path_cptac = args.dataDir+'CPTAC/multimodal_diag_survival_CPTAC.csv'
        excel_label_wsi_cptac = pd.read_csv(labels_path_cptac, header=0)
        # check_path_cptac = ('./data/CPTAC.xlsx')
        # check_wsi_cptac = pd.read_excel(check_path_cptac, header=0)
        # excel_label_wsi_cptac = excel_label_wsi_cptac[excel_label_wsi_cptac['WSI_ID'].isin(check_wsi_cptac['WSI_ID'].values)]
        excel_wsi_cptac = excel_label_wsi_cptac.values
        
        test_dataset_CPTAC = CPTAC_Dataset(excel_wsi=excel_wsi_cptac, args=args)
        test_dataset = test_dataset_CPTAC

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        loaders = (None, test_loader)  # trainloader is not needed for testing

        # for individually testing two datasets:
        # val_dataset_IvYGAP = IvYGAP_Dataset(phase='Val', args=args)
        # val_dataset_TCGA = TCGA_Dataset(phase='Val', args=args)
        # valtest_dataset_IvYGAP = torch.utils.data.ConcatDataset([val_dataset_IvYGAP, test_dataset_IvYGAP])
        # valtest_dataset_TCGA = torch.utils.data.ConcatDataset([val_dataset_TCGA, test_dataset_TCGA])
        # valtest_dataset_two = torch.utils.data.ConcatDataset([valtest_dataset_IvYGAP, valtest_dataset_TCGA])
        # loader_IvYGAP = DataLoader(valtest_dataset_IvYGAP, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        # loader_TCGA = DataLoader(valtest_dataset_TCGA, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        # loader_two = DataLoader(valtest_dataset_two, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        # loaders = (None, loader_TCGA)
        
        # Initialize model
        if args.mode == 'distillation':
            model, teacher_model = define_net(args)
            model = model.cuda()
            teacher_model = teacher_model.cuda()
        else:
            model = define_net(args).cuda()

        # model = define_net(args).cuda()

        model_fp = "#"

        model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
        if os.path.isfile(model_fp):
            model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
        else:
            raise FileNotFoundError("testing model not found at {}".format(model_fp))
            
        model = model.to(args.device)

        if args.mode == 'teacher':
            testTeachersModel(model, loaders, wandb_logger, args)
        elif args.mode == 'student':
            testStudentsModel(model, loaders, wandb_logger, args)
        elif args.mode == 'distillation':
            # testDistillation(model, teacher_model, loaders, wandb_logger, args)
            testStudentsModel(model, loaders, wandb_logger, args)
        elif args.mode == 'deformpathomic':
            testDeformPathomicModel(model, loaders, wandb_logger, args)
        elif args.mode == 'multiscale':
            testMultiScaleModel(model, loaders, wandb_logger, args)
        else:
            testBaselineModel(model, loaders, wandb_logger, args)

    elif args.coTraining: # implemented not testing yet!!!
        print("Now Inferencing TCGA & IvYGAP!")
        labels_path_tcga = args.dataDir+'TCGA/multimodal_diag_survival_TCGA.csv'
        excel_label_wsi_tcga = pd.read_csv(labels_path_tcga, header=0)
        excel_wsi_tcga = excel_label_wsi_tcga.values

        PATIENT_LIST_TCGA = excel_wsi_tcga[:,0]
        PATIENT_LIST_TCGA = list(PATIENT_LIST_TCGA)

        PATIENT_LIST_TCGA = np.unique(PATIENT_LIST_TCGA) #unique PATIENT_LIST: 952
        np.random.shuffle(PATIENT_LIST_TCGA)
        # print('unique len(PATIENT_LIST)', len(PATIENT_LIST)) #645
        NUM_PATIENT_ALL_TCGA = len(PATIENT_LIST_TCGA) # 952; 645 for mmd; 8:1:1= 1472:191:165

        # for IvYGAP:
        labels_path_ivygap = args.dataDir+'IvYGAP/multimodal_diag_survival_IvY.csv'
        excel_label_wsi_ivygap = pd.read_csv(labels_path_ivygap, header=0)
        excel_wsi_ivygap = excel_label_wsi_ivygap.values
        
        PATIENT_LIST_IVYGAP = excel_wsi_ivygap[:,0]
        PATIENT_LIST_IVYGAP = list(PATIENT_LIST_IVYGAP)

        PATIENT_LIST_IVYGAP = np.unique(PATIENT_LIST_IVYGAP) # unique PATIENT_LIST:33
        np.random.shuffle(PATIENT_LIST_IVYGAP)
        # print('unique len(PATIENT_LIST)', len(PATIENT_LIST)) #unique len(PATIENT_LIST) 33
        NUM_PATIENT_ALL_IVYGAP = len(PATIENT_LIST_IVYGAP) # 952; 645 for mmd; 8:1:1= 1472:191:165; 3:2 = 
        
        # K-Fold Spliting
        kfold_tcga  = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        kfold_ivygap = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

        for fold_tcga, (train_idx_tcga, test_idx_tcga) in enumerate(kfold_tcga.split(PATIENT_LIST_TCGA)):
            train_pid_tcga = PATIENT_LIST_TCGA[train_idx_tcga]
            test_pid_tcga = PATIENT_LIST_TCGA[test_idx_tcga]
            ## Control Fold / current fold here!! [0,1,2]
            if fold_tcga != 0:
                continue
            for fold_ivygap, (train_idx_ivygap, test_idx_ivygap) in enumerate(kfold_ivygap.split(PATIENT_LIST_IVYGAP)):
                train_pid_ivygap = PATIENT_LIST_IVYGAP[train_idx_ivygap]
                test_pid_ivygap = PATIENT_LIST_IVYGAP[test_idx_ivygap]
                
                if fold_ivygap == fold_tcga:
                    args.cur_fold = fold_ivygap
                    train_list_tcga = excel_wsi_tcga[np.isin(excel_wsi_tcga[:,0],train_pid_tcga)]
                    test_list_tcga = excel_wsi_tcga[np.isin(excel_wsi_tcga[:,0],test_pid_tcga)]
                    train_list_ivygap = excel_wsi_ivygap[np.isin(excel_wsi_ivygap[:,0],train_pid_ivygap)]
                    test_list_ivygap = excel_wsi_ivygap[np.isin(excel_wsi_ivygap[:,0],test_pid_ivygap)]
                    
                    # label related info
                    if args.printDataSplit:
                        # 1. 创建 DataFrame 
                        df = pd.DataFrame(train_list_tcga)
                        # 计算新列的值 Diag 2021
                        def calculate_new_column(row):#df.iloc
                            if row.iloc[4]=='WT':
                                return 0 #Grade4 GBM
                            elif row.iloc[5] == 'codel':
                                return 3 #Grade 2/3 Oligo
                            else:
                                if row.iloc[6] == -2 or row.iloc[6] == -1 or row.iloc[3] =='G4':
                                    return 1 #Grade 4 Astro
                                else:
                                    return 2 #Grade 2/3 Astro
                        df[len(df.columns)] = df.apply(calculate_new_column, axis=1)
                        os.makedirs('./outputs', exist_ok=True)
                        out_filename = './outputs/output_ivygap_train.csv'
                        if not os.path.exists(out_filename):
                            df.to_csv(out_filename, index=False, header=False)
                            print(f'文件 {out_filename} 创建并保存数据。')
                        else:
                            print(f'文件 {out_filename} 已存在。')

                        df = pd.DataFrame(test_list_tcga)
                        df[len(df.columns)] = df.apply(calculate_new_column, axis=1)
                        out_filename = './outputs/output_ivygap_test.csv'
                        if not os.path.exists(out_filename):
                            df.to_csv(out_filename, index=False, header=False)
                            print(f'文件 {out_filename} 创建并保存数据。')
                        else:
                            print(f'文件 {out_filename} 已存在。')                


                    # training set
                    if args.coTraining:
                        train_dataset_IvYGAP = IvYGAP_Dataset(excel_wsi=train_list_ivygap, args=args)

                    input_size_omic_IvYGAP = args.input_size_omic
                    input_size_omic_tumor_IvYGAP = args.input_size_omic_tumor
                    input_size_omic_immune_IvYGAP = args.input_size_omic_immune
                    
                    train_dataset_TCGA = TCGA_Dataset(excel_wsi=train_list_tcga, args=args)

                    input_size_omic_TCGA = args.input_size_omic
                    input_size_omic_tumor_TCGA = args.input_size_omic_tumor
                    input_size_omic_immune_TCGA = args.input_size_omic_immune
                        
                    input_size_omic = args.input_size_omic
                    input_size_omic_tumor = args.input_size_omic_tumor
                    input_size_omic_immune = args.input_size_omic_immune
                    
                    if args.coTraining:
                        train_dataset = torch.utils.data.ConcatDataset([train_dataset_IvYGAP, train_dataset_TCGA])
                    else:
                        train_dataset = train_dataset_TCGA

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=args.workers,
                    )

                    test_dataset_TCGA = TCGA_Dataset(excel_wsi=test_list_tcga, args=args)
                    if args.coTraining:
                        test_dataset_IvYGAP = IvYGAP_Dataset(excel_wsi=test_list_ivygap, args=args)
                        test_dataset = torch.utils.data.ConcatDataset([test_dataset_IvYGAP, test_dataset_TCGA])
                    else:
                        test_dataset = test_dataset_TCGA
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

                    loaders = (train_loader, test_loader)

                    step_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)
                    print(f'{step_per_epoch}')

                    # print('length of trainloader:', len(train_loader)) #36, 36 for 2 gpus; 73, 9, 10 for 1 gpus; batch_size=20
                    # print('length of testloader:', len(test_loader))   #10

                    # # model init
                    if args.mode == 'distillation':
                        model, teacher_model = define_net(args)
                        model = model.cuda()
                        teacher_model = teacher_model.cuda()
                    else:
                        model = define_net(args).cuda()

                    # reload model TCGA
                    # model_fp = args.checkpoints_teacher
                    model_fp = "#"
                    if os.path.isfile(model_fp):
                        model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
                    else:
                        raise FileNotFoundError("testing model not found at {}".format(model_fp))
             
                    model = model.to(args.device)

                    if args.mode == 'teacher':
                        testTeachersModel(model, loaders, wandb_logger, args)
                    elif args.mode == 'student':
                        testStudentsModel(model, loaders, wandb_logger, args)
                    elif args.mode == 'distillation':
                        # testDistillation(model, teacher_model, loaders, wandb_logger, args)
                        testStudentsModel(model, loaders, wandb_logger, args)
                    else: # args.mode == 'deformpathomic'
                        testDeformPathomicModel(model, loaders, wandb_logger, args)

    else: # implemented not testing yet!!!
        print("Now Inferencing TCGA!")
        labels_path_tcga = args.dataDir+'TCGA/multimodal_diag_survival_TCGA.csv'
        excel_label_wsi_tcga = pd.read_csv(labels_path_tcga, header=0)
        excel_wsi_tcga = excel_label_wsi_tcga.values

        PATIENT_LIST_TCGA = excel_wsi_tcga[:,0]
        PATIENT_LIST_TCGA = list(PATIENT_LIST_TCGA)

        PATIENT_LIST_TCGA = np.unique(PATIENT_LIST_TCGA) #unique PATIENT_LIST: 952
        np.random.shuffle(PATIENT_LIST_TCGA)
        # print('unique len(PATIENT_LIST)', len(PATIENT_LIST)) #645
        NUM_PATIENT_ALL_TCGA = len(PATIENT_LIST_TCGA) # 952; 645 for mmd; 8:1:1= 1472:191:165

        # for IvYGAP:
        labels_path_ivygap = args.dataDir+'IvYGAP/multimodal_diag_survival_IvY.csv'
        excel_label_wsi_ivygap = pd.read_csv(labels_path_ivygap, header=0)
        excel_wsi_ivygap = excel_label_wsi_ivygap.values
        
        PATIENT_LIST_IVYGAP = excel_wsi_ivygap[:,0]
        PATIENT_LIST_IVYGAP = list(PATIENT_LIST_IVYGAP)

        PATIENT_LIST_IVYGAP = np.unique(PATIENT_LIST_IVYGAP) # unique PATIENT_LIST:33
        np.random.shuffle(PATIENT_LIST_IVYGAP)
        # print('unique len(PATIENT_LIST)', len(PATIENT_LIST)) #unique len(PATIENT_LIST) 33
        NUM_PATIENT_ALL_IVYGAP = len(PATIENT_LIST_IVYGAP) # 952; 645 for mmd; 8:1:1= 1472:191:165; 3:2 = 
        
        # K-Fold Spliting
        kfold_tcga  = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        kfold_ivygap = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

        for fold_tcga, (train_idx_tcga, test_idx_tcga) in enumerate(kfold_tcga.split(PATIENT_LIST_TCGA)):
            train_pid_tcga = PATIENT_LIST_TCGA[train_idx_tcga]
            test_pid_tcga = PATIENT_LIST_TCGA[test_idx_tcga]
            ## Control Fold / current fold here!! [0,1,2]
            if fold_tcga != 0:
                continue
            for fold_ivygap, (train_idx_ivygap, test_idx_ivygap) in enumerate(kfold_ivygap.split(PATIENT_LIST_IVYGAP)):
                train_pid_ivygap = PATIENT_LIST_IVYGAP[train_idx_ivygap]
                test_pid_ivygap = PATIENT_LIST_IVYGAP[test_idx_ivygap]
                
                if fold_ivygap == fold_tcga:
                    args.cur_fold = fold_ivygap
                    train_list_tcga = excel_wsi_tcga[np.isin(excel_wsi_tcga[:,0],train_pid_tcga)]
                    test_list_tcga = excel_wsi_tcga[np.isin(excel_wsi_tcga[:,0],test_pid_tcga)]
                    train_list_ivygap = excel_wsi_ivygap[np.isin(excel_wsi_ivygap[:,0],train_pid_ivygap)]
                    test_list_ivygap = excel_wsi_ivygap[np.isin(excel_wsi_ivygap[:,0],test_pid_ivygap)]
                    
                    # label related info
                    if args.printDataSplit:
                        # 1. 创建 DataFrame 
                        df = pd.DataFrame(train_list_tcga)
                        # 计算新列的值 Diag 2021
                        def calculate_new_column(row):#df.iloc
                            if row.iloc[4]=='WT':
                                return 0 #Grade4 GBM
                            elif row.iloc[5] == 'codel':
                                return 3 #Grade 2/3 Oligo
                            else:
                                if row.iloc[6] == -2 or row.iloc[6] == -1 or row.iloc[3] =='G4':
                                    return 1 #Grade 4 Astro
                                else:
                                    return 2 #Grade 2/3 Astro
                        df[len(df.columns)] = df.apply(calculate_new_column, axis=1)
                        os.makedirs('./outputs', exist_ok=True)
                        out_filename = './outputs/output_ivygap_train.csv'
                        if not os.path.exists(out_filename):
                            df.to_csv(out_filename, index=False, header=False)
                            print(f'文件 {out_filename} 创建并保存数据。')
                        else:
                            print(f'文件 {out_filename} 已存在。')

                        df = pd.DataFrame(test_list_tcga)
                        df[len(df.columns)] = df.apply(calculate_new_column, axis=1)
                        out_filename = './outputs/output_ivygap_test.csv'
                        if not os.path.exists(out_filename):
                            df.to_csv(out_filename, index=False, header=False)
                            print(f'文件 {out_filename} 创建并保存数据。')
                        else:
                            print(f'文件 {out_filename} 已存在。')                


                    # training set
                    if args.coTraining:
                        train_dataset_IvYGAP = IvYGAP_Dataset(excel_wsi=train_list_ivygap, args=args)

                    input_size_omic_IvYGAP = args.input_size_omic
                    input_size_omic_tumor_IvYGAP = args.input_size_omic_tumor
                    input_size_omic_immune_IvYGAP = args.input_size_omic_immune
                    
                    train_dataset_TCGA = TCGA_Dataset(excel_wsi=train_list_tcga, args=args)

                    input_size_omic_TCGA = args.input_size_omic
                    input_size_omic_tumor_TCGA = args.input_size_omic_tumor
                    input_size_omic_immune_TCGA = args.input_size_omic_immune
                        
                    input_size_omic = args.input_size_omic
                    input_size_omic_tumor = args.input_size_omic_tumor
                    input_size_omic_immune = args.input_size_omic_immune
                    
                    if args.coTraining:
                        train_dataset = torch.utils.data.ConcatDataset([train_dataset_IvYGAP, train_dataset_TCGA])
                    else:
                        train_dataset = train_dataset_TCGA

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.workers,
                    )

                    test_dataset_TCGA = TCGA_Dataset(excel_wsi=test_list_tcga, args=args)
                    if args.coTraining:
                        test_dataset_IvYGAP = IvYGAP_Dataset(excel_wsi=test_list_ivygap, args=args)
                        test_dataset = torch.utils.data.ConcatDataset([test_dataset_IvYGAP, test_dataset_TCGA])
                    else:
                        test_dataset = test_dataset_TCGA
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

                    loaders = (train_loader, test_loader)

                    step_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)
                    print(f'{step_per_epoch}')

                    # print('length of trainloader:', len(train_loader)) #36, 36 for 2 gpus; 73, 9, 10 for 1 gpus; batch_size=20
                    # print('length of testloader:', len(test_loader))   #10

                    # # model init
                    if args.mode == 'distillation':
                        model, teacher_model = define_net(args)
                        model = model.cuda()
                        teacher_model = teacher_model.cuda()
                    else:
                        model = define_net(args).cuda()

                    # reload model TCGA
                    # model_fp = args.checkpoints_teacher
                    model_fp = "./checkpoints_diag_teacher_all_coswarm1_adamw_2e-4/fold_1_epoch_10_AUC_0.969547_ACC_0.869668_Sens_0.781594_Spec_0.958962_F1_0.775597_.pth"
                    if os.path.isfile(model_fp):
                        model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
                    else:
                        raise FileNotFoundError("testing model not found at {}".format(model_fp))
             
                    model = model.to(args.device)

                    if args.mode == 'teacher':
                        testTeachersModel(model, loaders, wandb_logger, args)
                    elif args.mode == 'student':
                        testStudentsModel(model, loaders, wandb_logger, args)
                    elif args.mode == 'distillation':
                        # testDistillation(model, teacher_model, loaders, wandb_logger, args)
                        testStudentsModel(model, loaders, wandb_logger, args)
                    elif args.mode == 'deformpathomic':
                        testDeformPathomicModel(model, loaders, wandb_logger, args)
                    elif args.mode == 'multiscale':
                        testMultiScaleModel(model, loaders, wandb_logger, args)
                    else:
                        testBaselineModel(model, loaders, wandb_logger, args)



if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/config_mine_diag2021.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb if not in debug mode
    if not args.debug:
        wandb.login(key="#")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project="MultiScale_TMI25",
            notes="MultiScale_TMI25",
            tags=["TMI25", "MultiScale"],
            config=config
        )
    else:
        wandb_logger = None


    main(0, args, wandb_logger)
