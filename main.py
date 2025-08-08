import os
import torch
import wandb
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from data.dataset import *
from torch.utils.data import DataLoader
from train_test import trainDeformPathomicModel, trainTeachersModel, trainStudentsModel, trainDistillation
from utils.yaml_config_hook import yaml_config_hook
from utils.sync_batchnorm import convert_model
from models.model import define_net, define_scheduler, define_optimizer
from sklearn.model_selection import StratifiedKFold, KFold


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # for CPTAC:
    if args.external_eval:
        print("Now Training CPTAC!")
        labels_path_cptac = args.dataDir+'CPTAC/multimodal_diag_survival_CPTAC.csv'
        excel_label_wsi_cptac = pd.read_csv(labels_path_cptac, header=0)

        # check_path_cptac = ('./data/CPTAC.xlsx')
        # check_wsi_cptac = pd.read_excel(check_path_cptac, header=0)
        # # print('CPTAC.shape:', check_wsi.shape)
        # excel_label_wsi_cptac = excel_label_wsi_cptac[excel_label_wsi_cptac['WSI_ID'].isin(check_wsi_cptac['WSI_ID'].values)]

        excel_wsi_cptac = excel_label_wsi_cptac.values
        
        PATIENT_LIST_CPTAC = excel_wsi_cptac[:,0]
        PATIENT_LIST_CPTAC = list(PATIENT_LIST_CPTAC)

        PATIENT_LIST_CPTAC = np.unique(PATIENT_LIST_CPTAC) #unique PATIENT_LIST: 952
        np.random.shuffle(PATIENT_LIST_CPTAC)
        # print('unique len(PATIENT_LIST)', len(PATIENT_LIST)) #645
        NUM_PATIENT_ALL_CPTAC = len(PATIENT_LIST_CPTAC) # 952; 645 for mmd; 8:1:1= 1472:191:165
        
        # K-Fold Spliting
        kfold_cptac = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

        for fold_cptac, (train_idx_cptac, test_idx_cptac) in enumerate(kfold_cptac.split(PATIENT_LIST_CPTAC)):
            train_pid_cptac = PATIENT_LIST_CPTAC[train_idx_cptac]
            test_pid_cptac = PATIENT_LIST_CPTAC[test_idx_cptac]

            ## Control Fold / current fold here!! [0,1,2]
            # if fold_cptac != 0:
            #     continue

            args.cur_fold = fold_cptac
            train_list_cptac = excel_wsi_cptac[np.isin(excel_wsi_cptac[:,0],train_pid_cptac)]
            test_list_cptac = excel_wsi_cptac[np.isin(excel_wsi_cptac[:,0],test_pid_cptac)]
            
            if args.printDataSplit:
                # 1. 创建 DataFrame
                df = pd.DataFrame(train_list_cptac)
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
                out_filename = 'output_cptac_train.csv'
                if not os.path.exists(out_filename):
                    df.to_csv(out_filename, index=False, header=False)
                    print(f'文件 {out_filename} 创建并保存数据。')
                else:
                    print(f'文件 {out_filename} 已存在。')

                df = pd.DataFrame(test_list_cptac)
                df[len(df.columns)] = df.apply(calculate_new_column, axis=1)
                out_filename = 'output_cptac_test.csv'
                if not os.path.exists(out_filename):
                    df.to_csv(out_filename, index=False, header=False)
                    print(f'文件 {out_filename} 创建并保存数据。')
                else:
                    print(f'文件 {out_filename} 已存在。')                


            # training set
            train_dataset_CPTAC = CPTAC_Dataset(excel_wsi=train_list_cptac, args=args)
            args.input_size_omic = train_dataset_CPTAC.input_size_omic
            args.input_size_omic_tumor = train_dataset_CPTAC.input_size_omic_tumor
            args.input_size_omic_immune = train_dataset_CPTAC.input_size_omic_immune
            
            train_dataset = train_dataset_CPTAC

            # set sampler for parallel training
            if args.world_size > 1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
                )
            else:
                train_sampler = None

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                drop_last=True,
                num_workers=args.workers,
                sampler=train_sampler,
            )
            if rank == 0:
                test_dataset_CPTAC = CPTAC_Dataset(excel_wsi=test_list_cptac, args=args)
                test_dataset = test_dataset_CPTAC
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            else:
                test_loader = None
                val_loader = None

            loaders = (train_loader, test_loader)

            step_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)
            print(f'{step_per_epoch}')

            # print('length of trainloader:', len(train_loader)) #36, 36 for 2 gpus; 73, 9, 10 for 1 gpus; batch_size=20
            # print('length of valloader:', len(val_loader))     #9
            # print('length of testloader:', len(test_loader))   #10

            # # model init
            # model_tumor = define_net(args, input_size_omic_tumor).cuda()
            if args.mode == 'distillation':
                model, teacher_model = define_net(args)
                model = model.cuda()
                teacher_model = teacher_model.cuda()
            else:
                model = define_net(args).cuda()

            # reload model CPTAC
            if args.reload:
                # model_fp = os.path.join(
                #     args.checkpoints, "epoch_{}_.pth".format(args.epochs)
                # )
                # model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
                # model_fp = os.path.join(
                #     args.checkpoints, "best_modal.pth"
                # )
                if args.mode == 'teacher':
                    model_fp = args.checkpoints_teacher
                elif args.mode == 'student':
                    model_fp = args.checkpoints_student
                else:
                    model_fp = "#"
                model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
            
            if args.mode == 'distillation':
                model_fp = args.checkpoints_student
                model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
                
                teacher_model_fp = args.checkpoints_teacher
                teacher_model.load_state_dict(torch.load(teacher_model_fp, map_location='cuda:0'))
                
                
            model = model.to(args.device)

            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer, step_per_epoch)

            if args.dataparallel:
                model = convert_model(model)
                model = DataParallel(model)

            else:
                if args.world_size > 1:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
                    # model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
                    # model._set_static_graph()

            if args.mode == 'distillation':
                teacher_model = teacher_model.to(args.device)
                # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                optimizer_tea = define_optimizer(args, teacher_model)
                scheduler_tea = define_scheduler(args, optimizer_tea) #not used
                if args.dataparallel:
                    teacher_model = convert_model(teacher_model)
                    teacher_model = DataParallel(teacher_model)
                else:
                    if args.world_size > 1:
                        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
                        teacher_model = DDP(teacher_model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
                        # model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
                        # model._set_static_graph()

            if args.mode == 'deformpathomic':
                print("training deformpathomic model")
                trainDeformPathomicModel(model, loaders, optimizer, scheduler, wandb_logger, args)
            elif args.mode == 'teacher':
                print('training teacher model')
                trainTeachersModel(model, loaders, optimizer, scheduler, wandb_logger, args)   
            elif args.mode == 'student':
                print('training student model')
                trainStudentsModel(model, loaders, optimizer, scheduler, wandb_logger, args)
            else: # args.mode == 'distillation':
                print('fintuning student with distillation')
                trainDistillation(model, teacher_model, loaders, optimizer, scheduler, wandb_logger, args)


    # for TCGA and IvYGAP:
    else:
        print('Now is the TCGA and IvYGAP CoTraining!')
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
            # if fold_tcga != 0:
            #     continue
            for fold_ivygap, (train_idx_ivygap, test_idx_ivygap) in enumerate(kfold_ivygap.split(PATIENT_LIST_IVYGAP)):
                train_pid_ivygap = PATIENT_LIST_IVYGAP[train_idx_ivygap]
                test_pid_ivygap = PATIENT_LIST_IVYGAP[test_idx_ivygap]
                
                if fold_ivygap == fold_tcga:
                    args.cur_fold = fold_ivygap
                    train_list_tcga = excel_wsi_tcga[np.isin(excel_wsi_tcga[:,0],train_pid_tcga)]
                    test_list_tcga = excel_wsi_tcga[np.isin(excel_wsi_tcga[:,0],test_pid_tcga)]
                    train_list_ivygap = excel_wsi_ivygap[np.isin(excel_wsi_ivygap[:,0],train_pid_ivygap)]
                    test_list_ivygap = excel_wsi_ivygap[np.isin(excel_wsi_ivygap[:,0],test_pid_ivygap)]
                    

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
                        out_filename = 'output_ivygap_train.csv'
                        if not os.path.exists(out_filename):
                            df.to_csv(out_filename, index=False, header=False)
                            print(f'文件 {out_filename} 创建并保存数据。')
                        else:
                            print(f'文件 {out_filename} 已存在。')

                        df = pd.DataFrame(test_list_tcga)
                        df[len(df.columns)] = df.apply(calculate_new_column, axis=1)
                        out_filename = 'output_ivygap_test.csv'
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

                    # set sampler for parallel training
                    if args.world_size > 1:
                        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
                        )
                    else:
                        train_sampler = None

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        shuffle=(train_sampler is None),
                        drop_last=True,
                        num_workers=args.workers,
                        sampler=train_sampler,
                    )
                    if rank == 0:
                        test_dataset_TCGA = TCGA_Dataset(excel_wsi=test_list_tcga, args=args)
                        if args.coTraining:
                            test_dataset_IvYGAP = IvYGAP_Dataset(excel_wsi=test_list_ivygap, args=args)
                            test_dataset = torch.utils.data.ConcatDataset([test_dataset_IvYGAP, test_dataset_TCGA])
                        else:
                            test_dataset = test_dataset_TCGA
                        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                    else:
                        test_loader = None
                        val_loader = None

                    loaders = (train_loader, test_loader)

                    step_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)
                    print(f'{step_per_epoch}')

                    # # model init
                    # model_tumor = define_net(args, input_size_omic_tumor).cuda()
                    if args.mode == 'distillation':
                        model, teacher_model = define_net(args)
                        model = model.cuda()
                        teacher_model = teacher_model.cuda()
                    else:
                        model = define_net(args).cuda()

                    # reload model TCGA
                    if args.reload:
                        # model_fp = os.path.join(
                        #     args.checkpoints, "epoch_{}_.pth".format(args.epochs)
                        # )
                        # model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
                        # model_fp = os.path.join(
                        #     args.checkpoints, "best_modal.pth"
                        # )
                        model_fp = args.checkpoints_teacher
                        model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
                    
                    if args.mode == 'distillation':
                        model_fp = args.checkpoints_student
                        model.load_state_dict(torch.load(model_fp, map_location='cuda:0'))
                        
                        teacher_model_fp = args.checkpoints_teacher
                        teacher_model.load_state_dict(torch.load(teacher_model_fp, map_location='cuda:0'))
                        
                        
                    model = model.to(args.device)

                    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    optimizer = define_optimizer(args, model)
                    scheduler = define_scheduler(args, optimizer, step_per_epoch)

                    if args.dataparallel:
                        model = convert_model(model)
                        model = DataParallel(model)

                    else:
                        if args.world_size > 1:
                            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                            model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
                            # model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
                            # model._set_static_graph()

                    if args.mode == 'distillation':
                        teacher_model = teacher_model.to(args.device)
                        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        optimizer_tea = define_optimizer(args, teacher_model)
                        scheduler_tea = define_scheduler(args, optimizer_tea) #not used
                        if args.dataparallel:
                            teacher_model = convert_model(teacher_model)
                            teacher_model = DataParallel(teacher_model)
                        else:
                            if args.world_size > 1:
                                teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
                                teacher_model = DDP(teacher_model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
                                # model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
                                # model._set_static_graph()

                    if args.mode == 'deformpathomic':
                        print("training deformpathomic model")
                        trainDeformPathomicModel(model, loaders, optimizer, scheduler, wandb_logger, args)
                    elif args.mode == 'teacher':
                        print('training teacher model')
                        trainTeachersModel(model, loaders, optimizer, scheduler, wandb_logger, args)    
                    elif args.mode == 'student':
                        print('training student model')
                        trainStudentsModel(model, loaders, optimizer, scheduler, wandb_logger, args)
                    else: #args.mode == 'distillation':
                        print('fintuning student with distillation')
                        trainDistillation(model, teacher_model, loaders, optimizer, scheduler, wandb_logger, args)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/config_mine_diag2021.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    # Master address for distributed data parallel
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visiable_device
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

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


    if args.world_size > 1:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger,), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger)
