import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, precision_score

from sklearn.metrics import auc as aucFunc
from sklearn.metrics import roc_curve

from utils.utils import CIndex_sksurv
from imblearn.metrics import sensitivity_score, specificity_score

def make_one_hot(data1,N=0):
    if N!=0:
        num=N
    else:
        num = int(np.max(data1) + 1)
    return (np.arange(num)==data1[:,None]).astype(np.int16)

def compute_avg_metrics_micro(groundTruth, activations):
    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    
    multi_class = 'ovr'
    avg = 'micro' #'macro'
    mean_acc = accuracy_score(y_true=groundTruth, y_pred=predictions)
    
    # For binary classification
    xf_activations = activations

    if activations.shape[1] == 2:
        activations = activations[:, 1]
        multi_class = 'raise'
        avg = 'binary'
    try:
        # print("Unique classes in y_true:", len(np.unique(groundTruth))) #4
        # print("Shape of y_score:", activations.shape) #(172,4)
        if activations.shape[-1] < 2:
            auc = roc_auc_score(y_true=groundTruth, y_score=activations, multi_class=multi_class)
        elif activations.shape[-1] > 2:
            auc = roc_auc_score(y_true=groundTruth, y_score=activations, multi_class=multi_class, average='micro')
            # print('diag roc_auc_score ovr micro auc:', auc)
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    f1_macro = f1_score(y_true=groundTruth, y_pred=predictions, average=avg)
    bac = balanced_accuracy_score(y_true=groundTruth, y_pred=predictions)
    sens = sensitivity_score(y_true=groundTruth, y_pred=predictions, average=avg)
    spec = specificity_score(y_true=groundTruth, y_pred=predictions, average=avg)
    prec = precision_score(y_true=groundTruth, y_pred=predictions, average=avg)

    # xf_AUC >2:
    if xf_activations.shape[-1] > 2:
        label_tiantan_Ours_onehot=make_one_hot(groundTruth,N=4)
        pre_list_Ours = xf_activations.ravel()
        label_list_Ours =label_tiantan_Ours_onehot.ravel()
        fpr_micro, tpr_micro, _ = roc_curve(label_list_Ours, pre_list_Ours)
        micro_auc_Ours = aucFunc(fpr_micro, tpr_micro)
        print('yp_auc vs xf_auc:', auc, micro_auc_Ours)
    elif xf_activations.shape[-1] == 2:
        fpr_micro, tpr_micro, _ = roc_curve(groundTruth, activations)
        micro_auc_Ours = aucFunc(fpr_micro, tpr_micro)
        print('yp_auc vs xf_auc:', auc, micro_auc_Ours)

    return mean_acc, f1_macro, auc, bac, sens, spec, prec
    # return mean_acc, f1_macro, micro_auc_Ours, bac, sens_macro, spec_macro, prec_macro

def compute_avg_metrics(groundTruth, activations):
    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    mean_acc = accuracy_score(y_true=groundTruth, y_pred=predictions)
    f1_macro = f1_score(y_true=groundTruth, y_pred=predictions, average='macro')
    try:
        auc = roc_auc_score(y_true=groundTruth, y_score=activations, multi_class='ovr')
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    bac = balanced_accuracy_score(y_true=groundTruth, y_pred=predictions)
    sens_macro = sensitivity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    spec_macro = specificity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    prec_macro = precision_score(y_true=groundTruth, y_pred=predictions, average='macro')

    return mean_acc, f1_macro, auc, bac, sens_macro, spec_macro, prec_macro


def compute_confusion_matrix(groundTruth, activations, labels):

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions, labels=labels)

    return cm

# for teacher student model
def epochDistillVal(model, dataLoader, args, model_type='teacher'):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    if model_type == 'student':
        activations_stu10 = torch.Tensor().cuda()
        activations_stu20 = torch.Tensor().cuda()
        activations_stus = torch.Tensor().cuda()
    elif model_type == 'teacher':
        activations_tea10 = torch.Tensor().cuda()
        activations_tea20 = torch.Tensor().cuda()
        activations_teas = torch.Tensor().cuda()

    with torch.no_grad():
        # for i, (x_path, x_omic, label) in enumerate(dataLoader):
            # x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
            # fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
            # output = F.softmax(logits, dim=1)
            # groundTruth = torch.cat((groundTruth, label[:, 5]))
            # activations = torch.cat((activations, output))
            # 0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time
        if args.save4visualization:
            for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label, wsiID) in enumerate(dataLoader):
                x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
                if model_type == 'teacher':
                    feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
                elif model_type == 'student':
                    feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20)

                if args.task_type == "diag2021":
                    groundTruth = torch.cat((groundTruth, label[:, 5]))
                elif args.task_type == "grade":
                    groundTruth = torch.cat((groundTruth, label[:, 4]))
                elif args.task_type == "subtype":
                    groundTruth = torch.cat((groundTruth, label[:, 7]))
                
                if model_type == 'student':
                    output_stu10 = F.softmax(logits_dict['logits_stu10'], dim=1)
                    activations_stu10 = torch.cat((activations_stu10, output_stu10))

                    output_stu20 = F.softmax(logits_dict['logits_stu20'], dim=1)
                    activations_stu20 = torch.cat((activations_stu20, output_stu20))

                    output_stus = F.softmax(logits_dict['logits_stus'], dim=1)
                    activations_stus = torch.cat((activations_stus, output_stus))

                elif model_type == 'teacher':
                    output_tea10 = F.softmax(logits_dict['logits_tea10'], dim=1)
                    activations_tea10 = torch.cat((activations_tea10, output_tea10))

                    output_tea20 = F.softmax(logits_dict['logits_tea20'], dim=1)
                    activations_tea20 = torch.cat((activations_tea20, output_tea20))

                    output_teas = F.softmax(logits_dict['logits_teas'], dim=1)
                    activations_teas = torch.cat((activations_teas, output_teas))        
        else:
            for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):   
                x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
                if model_type == 'teacher':
                    feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
                elif model_type == 'student':
                    feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20)

                if args.task_type == "diag2021":
                    groundTruth = torch.cat((groundTruth, label[:, 5]))
                elif args.task_type == "grade":
                    groundTruth = torch.cat((groundTruth, label[:, 4]))
                elif args.task_type == "subtype":
                    groundTruth = torch.cat((groundTruth, label[:, 7]))
                
                if model_type == 'student':
                    output_stu10 = F.softmax(logits_dict['logits_stu10'], dim=1)
                    activations_stu10 = torch.cat((activations_stu10, output_stu10))

                    output_stu20 = F.softmax(logits_dict['logits_stu20'], dim=1)
                    activations_stu20 = torch.cat((activations_stu20, output_stu20))

                    output_stus = F.softmax(logits_dict['logits_stus'], dim=1)
                    activations_stus = torch.cat((activations_stus, output_stus))

                elif model_type == 'teacher':
                    output_tea10 = F.softmax(logits_dict['logits_tea10'], dim=1)
                    activations_tea10 = torch.cat((activations_tea10, output_tea10))

                    output_tea20 = F.softmax(logits_dict['logits_tea20'], dim=1)
                    activations_tea20 = torch.cat((activations_tea20, output_tea20))

                    output_teas = F.softmax(logits_dict['logits_teas'], dim=1)
                    activations_teas = torch.cat((activations_teas, output_teas))

        if model_type == 'student': 
            acc_stu10, f1_stu10, auc_stu10, bac_stu10, sens_stu10, spec_stu10, prec_stu10 = compute_avg_metrics(groundTruth, activations_stu10)
            acc_stu20, f1_stu20, auc_stu20, bac_stu20, sens_stu20, spec_stu20, prec_stu20 = compute_avg_metrics(groundTruth, activations_stu20)
            acc_stus, f1_stus, auc_stus, bac_stus, sens_stus, spec_stus, prec_stus = compute_avg_metrics(groundTruth, activations_stus)

            acc_dict = {'acc_stu10': acc_stu10,
                        'acc_stu20': acc_stu20,
                        'acc_stus': acc_stus,
                        }
            f1_dict = {'f1_stu10': f1_stu10,
                        'f1_stu20': f1_stu20,
                        'f1_stus': f1_stus,
                        }
            auc_dict = {'auc_stu10': auc_stu10,
                        'auc_stu20': auc_stu20,
                        'auc_stus': auc_stus,
                        }
            bac_dict = {'bac_stu10': bac_stu10,
                        'bac_stu20': bac_stu20,
                        'bac_stus': bac_stus,
                        }
            sens_dict = {'sens_stu10': sens_stu10,
                        'sens_stu20': sens_stu20,
                        'sens_stus': sens_stus,
                        }
            spec_dict = {'spec_stu10': spec_stu10,
                        'spec_stu20': spec_stu20,
                        'spec_stus': spec_stus,
                        }
            prec_dict = {'prec_stu10': prec_stu10,
                        'prec_stu20': prec_stu20,
                        'prec_stus': prec_stus,
                        }

        elif model_type == 'teacher':
            acc_tea10, f1_tea10, auc_tea10, bac_tea10, sens_tea10, spec_tea10, prec_tea10 = compute_avg_metrics(groundTruth, activations_tea10)
            acc_tea20, f1_tea20, auc_tea20, bac_tea20, sens_tea20, spec_tea20, prec_tea20 = compute_avg_metrics(groundTruth, activations_tea20)
            acc_teas, f1_teas, auc_teas, bac_teas, sens_teas, spec_teas, prec_teas = compute_avg_metrics(groundTruth, activations_teas)

            acc_dict = {'acc_teas': acc_teas,
                        'acc_tea10': acc_tea10,
                        'acc_tea20': acc_tea20,
                        }
            f1_dict = {'f1_teas': f1_teas,
                        'f1_tea10': f1_tea10,
                        'f1_tea20': f1_tea20,
                        }
            auc_dict = {'auc_teas': auc_teas,
                        'auc_tea10': auc_tea10,
                        'auc_tea20': auc_tea20,
                        }
            bac_dict = {'bac_teas': bac_teas,
                        'bac_tea10': bac_tea10,
                        'bac_tea20': bac_tea20,
                        }
            sens_dict = {'sens_teas': sens_teas,
                        'sens_tea10': sens_tea10,
                        'sens_tea20': sens_tea20,
                        }
            spec_dict = {'spec_teas': spec_teas,
                        'spec_tea10': spec_tea10,
                        'spec_tea20': spec_tea20,
                        }
            prec_dict = {'prec_teas': prec_teas,
                        'prec_tea10': prec_tea10,
                        'prec_tea20': prec_tea20,
                        }

    model.train(training)

    return acc_dict, f1_dict, auc_dict, bac_dict, sens_dict, spec_dict, prec_dict

def epochDistillVal_survival(model, dataLoader, args, model_type='teacher'):
    training = model.training
    model.eval()

    # groundTruth = torch.Tensor().cuda()
    # activations = torch.Tensor().cuda()
    if model_type=='student':
        risk_pred_all_stu10, censor_all_stu10, survtime_all_stu10 = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        risk_pred_all_stu20, censor_all_stu20, survtime_all_stu20 = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        risk_pred_all_stus, censor_all_stus, survtime_all_stus = np.array([]), np.array([]), np.array([])       # Used for calculating the C-Index
    elif model_type=='teacher':
        risk_pred_all_tea10, censor_all_tea10, survtime_all_tea10 = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        risk_pred_all_tea20, censor_all_tea20, survtime_all_tea20 = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        risk_pred_all_teas, censor_all_teas, survtime_all_teas = np.array([]), np.array([]), np.array([])       # Used for calculating the C-Index
    with torch.no_grad():
        for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            if model_type == 'teacher':
                feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            elif model_type == 'student':
                feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20)
            # logits:[hazard_tumor, hazard_immune, hazard, omic_tumor, vgrid_tumor, omic_immune, vgrid_immune]
            # 0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time
            
            if model_type == 'student':
                risk_pred_all_stu10 = np.concatenate((risk_pred_all_stu10, risk_dict['risk_stu10'].detach().cpu().numpy().reshape(-1)))
                censor_all_stu10 = np.concatenate((censor_all_stu10, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all_stu10 = np.concatenate((survtime_all_stu10, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

                risk_pred_all_stu20 = np.concatenate((risk_pred_all_stu20, risk_dict['risk_stu20'].detach().cpu().numpy().reshape(-1)))
                censor_all_stu20 = np.concatenate((censor_all_stu20, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all_stu20 = np.concatenate((survtime_all_stu20, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

                # risk_stus
                risk_pred_all_stus = np.concatenate((risk_pred_all_stus, risk_dict['risk_stus'].detach().cpu().numpy().reshape(-1)))
                censor_all_stus = np.concatenate((censor_all_stus, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all_stus = np.concatenate((survtime_all_stus, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

            elif model_type == 'teacher':
                risk_pred_all_tea10 = np.concatenate((risk_pred_all_tea10, risk_dict['risk_tea10'].detach().cpu().numpy().reshape(-1)))
                censor_all_tea10 = np.concatenate((censor_all_tea10, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all_tea10 = np.concatenate((survtime_all_tea10, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

                risk_pred_all_tea20 = np.concatenate((risk_pred_all_tea20, risk_dict['risk_tea20'].detach().cpu().numpy().reshape(-1)))
                censor_all_tea20 = np.concatenate((censor_all_tea20, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all_tea20 = np.concatenate((survtime_all_tea20, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

                # risk_stus
                risk_pred_all_teas = np.concatenate((risk_pred_all_teas, risk_dict['risk_teas'].detach().cpu().numpy().reshape(-1)))
                censor_all_teas = np.concatenate((censor_all_teas, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all_teas = np.concatenate((survtime_all_teas, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information
        # print('risk_pred_all.shape:', risk_pred_all.shape) #236
        # print('event_all.shape:', event_all.shape) #236
        if model_type == 'student':
            cindex_stu10 = CIndex_sksurv(all_risk_scores=risk_pred_all_stu10, all_censorships=censor_all_stu10, all_event_times=survtime_all_stu10)
            cindex_stu20 = CIndex_sksurv(all_risk_scores=risk_pred_all_stu20, all_censorships=censor_all_stu20, all_event_times=survtime_all_stu20)
            cindex_stus = CIndex_sksurv(all_risk_scores=risk_pred_all_stus, all_censorships=censor_all_stus, all_event_times=survtime_all_stus)
            cindex_dict = {'cindex_stu10': cindex_stu10,
                                'cindex_stu20': cindex_stu20,
                                'cindex_stus': cindex_stus,
                                }
        elif model_type == 'teacher':
            cindex_tea10 = CIndex_sksurv(all_risk_scores=risk_pred_all_tea10, all_censorships=censor_all_tea10, all_event_times=survtime_all_tea10)
            cindex_tea20 = CIndex_sksurv(all_risk_scores=risk_pred_all_tea20, all_censorships=censor_all_tea20, all_event_times=survtime_all_tea20)
            cindex_teas = CIndex_sksurv(all_risk_scores=risk_pred_all_teas, all_censorships=censor_all_teas, all_event_times=survtime_all_teas)
            cindex_dict = {'cindex_tea10': cindex_tea10,
                                'cindex_tea20': cindex_tea20,
                                'cindex_teas': cindex_teas,
                                }

    model.train(training)

    return cindex_dict

# for multiscale deformclustermerge model
def epochScalesVal(model, dataLoader, args):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations_stu10 = torch.Tensor().cuda()
    activations_stu20 = torch.Tensor().cuda()
    activations_stus = torch.Tensor().cuda()
    activations_tea10 = torch.Tensor().cuda()
    activations_tea20 = torch.Tensor().cuda()

    with torch.no_grad():
        # for i, (x_path, x_omic, label) in enumerate(dataLoader):
            # x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
            # fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
            # output = F.softmax(logits, dim=1)
            # groundTruth = torch.cat((groundTruth, label[:, 5]))
            # activations = torch.cat((activations, output))
            # 0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time
        for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            

            if args.task_type == "diag2021":
                groundTruth = torch.cat((groundTruth, label[:, 5]))
            elif args.task_type == "grade":
                groundTruth = torch.cat((groundTruth, label[:, 4]))
            elif args.task_type == "subtype":
                groundTruth = torch.cat((groundTruth, label[:, 7]))

            output_stu10 = F.softmax(logits_dict['logits_stu10'], dim=1)
            activations_stu10 = torch.cat((activations_stu10, output_stu10))

            output_stu20 = F.softmax(logits_dict['logits_stu20'], dim=1)
            activations_stu20 = torch.cat((activations_stu20, output_stu20))

            output_stus = F.softmax(logits_dict['logits_stus'], dim=1)
            activations_stus = torch.cat((activations_stus, output_stus))

            output_tea10 = F.softmax(logits_dict['logits_tea10'], dim=1)
            activations_tea10 = torch.cat((activations_tea10, output_tea10))

            output_tea20 = F.softmax(logits_dict['logits_tea20'], dim=1)
            activations_tea20 = torch.cat((activations_tea20, output_tea20))
            
        acc_stu10, f1_stu10, auc_stu10, bac_stu10, sens_stu10, spec_stu10, prec_stu10 = compute_avg_metrics(groundTruth, activations_stu10)
        acc_stu20, f1_stu20, auc_stu20, bac_stu20, sens_stu20, spec_stu20, prec_stu20 = compute_avg_metrics(groundTruth, activations_stu20)
        acc_stus, f1_stus, auc_stus, bac_stus, sens_stus, spec_stus, prec_stus = compute_avg_metrics(groundTruth, activations_stus)
        acc_tea10, f1_tea10, auc_tea10, bac_tea10, sens_tea10, spec_tea10, prec_tea10 = compute_avg_metrics(groundTruth, activations_tea10)
        acc_tea20, f1_tea20, auc_tea20, bac_tea20, sens_tea20, spec_tea20, prec_tea20 = compute_avg_metrics(groundTruth, activations_tea20)

        acc_dict = {'acc_stu10': acc_stu10,
                    'acc_stu20': acc_stu20,
                    'acc_stus': acc_stus,
                    'acc_tea10': acc_tea10,
                    'acc_tea20': acc_tea20,
                    }
        f1_dict = {'f1_stu10': f1_stu10,
                    'f1_stu20': f1_stu20,
                    'f1_stus': f1_stus,
                    'f1_tea10': f1_tea10,
                    'f1_tea20': f1_tea20,
                    }
        auc_dict = {'auc_stu10': auc_stu10,
                    'auc_stu20': auc_stu20,
                    'auc_stus': auc_stus,
                    'auc_tea10': auc_tea10,
                    'auc_tea20': auc_tea20,
                    }
        bac_dict = {'bac_stu10': bac_stu10,
                    'bac_stu20': bac_stu20,
                    'bac_stus': bac_stus,
                    'bac_tea10': bac_tea10,
                    'bac_tea20': bac_tea20,
                    }
        sens_dict = {'sens_stu10': sens_stu10,
                    'sens_stu20': sens_stu20,
                    'sens_stus': sens_stus,
                    'sens_tea10': sens_tea10,
                    'sens_tea20': sens_tea20,
                    }
        spec_dict = {'spec_stu10': spec_stu10,
                    'spec_stu20': spec_stu20,
                    'spec_stus': spec_stus,
                    'spec_tea10': spec_tea10,
                    'spec_tea20': spec_tea20,
                    }
        prec_dict = {'prec_stu10': prec_stu10,
                    'prec_stu20': prec_stu20,
                    'prec_stus': prec_stus,
                    'prec_tea10': prec_tea10,
                    'prec_tea20': prec_tea20,
                    }

    model.train(training)

    return acc_dict, f1_dict, auc_dict, bac_dict, sens_dict, spec_dict, prec_dict

def epochScalesVal_survival(model, dataLoader, args):
    training = model.training
    model.eval()

    # groundTruth = torch.Tensor().cuda()
    # activations = torch.Tensor().cuda()
    
    risk_pred_all_stu10, censor_all_stu10, survtime_all_stu10 = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
    risk_pred_all_stu20, censor_all_stu20, survtime_all_stu20 = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
    risk_pred_all_stus, censor_all_stus, survtime_all_stus = np.array([]), np.array([]), np.array([])       # Used for calculating the C-Index
    risk_pred_all_tea10, censor_all_tea10, survtime_all_tea10 = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
    risk_pred_all_tea20, censor_all_tea20, survtime_all_tea20 = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
    with torch.no_grad():
        for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            # logits:[hazard_tumor, hazard_immune, hazard, omic_tumor, vgrid_tumor, omic_immune, vgrid_immune]
            # 0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time
            
            risk_pred_all_stu10 = np.concatenate((risk_pred_all_stu10, risk_dict['risk_stu10'].detach().cpu().numpy().reshape(-1)))
            censor_all_stu10 = np.concatenate((censor_all_stu10, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all_stu10 = np.concatenate((survtime_all_stu10, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

            risk_pred_all_stu20 = np.concatenate((risk_pred_all_stu20, risk_dict['risk_stu20'].detach().cpu().numpy().reshape(-1)))
            censor_all_stu20 = np.concatenate((censor_all_stu20, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all_stu20 = np.concatenate((survtime_all_stu20, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

            # risk_stus
            risk_pred_all_stus = np.concatenate((risk_pred_all_stus, risk_dict['risk_stus'].detach().cpu().numpy().reshape(-1)))
            censor_all_stus = np.concatenate((censor_all_stus, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all_stus = np.concatenate((survtime_all_stus, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

            risk_pred_all_tea10 = np.concatenate((risk_pred_all_tea10, risk_dict['risk_tea10'].detach().cpu().numpy().reshape(-1)))
            censor_all_tea10 = np.concatenate((censor_all_tea10, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all_tea10 = np.concatenate((survtime_all_tea10, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

            risk_pred_all_tea20 = np.concatenate((risk_pred_all_tea20, risk_dict['risk_tea20'].detach().cpu().numpy().reshape(-1)))
            censor_all_tea20 = np.concatenate((censor_all_tea20, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all_tea20 = np.concatenate((survtime_all_tea20, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information

        
        # print('risk_pred_all.shape:', risk_pred_all.shape) #236
        # print('event_all.shape:', event_all.shape) #236

        cindex_stu10 = CIndex_sksurv(all_risk_scores=risk_pred_all_stu10, all_censorships=censor_all_stu10, all_event_times=survtime_all_stu10)
        cindex_stu20 = CIndex_sksurv(all_risk_scores=risk_pred_all_stu20, all_censorships=censor_all_stu20, all_event_times=survtime_all_stu20)
        cindex_stus = CIndex_sksurv(all_risk_scores=risk_pred_all_stus, all_censorships=censor_all_stus, all_event_times=survtime_all_stus)
        cindex_tea10 = CIndex_sksurv(all_risk_scores=risk_pred_all_tea10, all_censorships=censor_all_tea10, all_event_times=survtime_all_tea10)
        cindex_tea20 = CIndex_sksurv(all_risk_scores=risk_pred_all_tea20, all_censorships=censor_all_tea20, all_event_times=survtime_all_tea20)

        cindex_dict = {'cindex_stu10': cindex_stu10,
                             'cindex_stu20': cindex_stu20,
                             'cindex_stus': cindex_stus,
                             'cindex_tea10': cindex_tea10,
                             'cindex_tea20': cindex_tea20,
                             }

    model.train(training)

    return cindex_dict

# for deformpathomic model
def epochVal(model, dataLoader, args):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        # for i, (x_path, x_omic, label) in enumerate(dataLoader):
            # x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
            # fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
            # output = F.softmax(logits, dim=1)
            # groundTruth = torch.cat((groundTruth, label[:, 5]))
            # activations = torch.cat((activations, output))
        for i, (x_path, _, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            output = F.softmax(logits[2], dim=1)
            if args.task_type == "diag2021":
                groundTruth = torch.cat((groundTruth, label[:, 5]))
            elif args.task_type == "grade":
                groundTruth = torch.cat((groundTruth, label[:, 4]))
            elif args.task_type == "subtype":
                groundTruth = torch.cat((groundTruth, label[:, 7]))
            activations = torch.cat((activations, output))
            
        acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return acc, f1, auc, bac, sens, spec, prec

def epochVal_survival(model, dataLoader, args):
    training = model.training
    model.eval()

    # groundTruth = torch.Tensor().cuda()
    # activations = torch.Tensor().cuda()
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])   # Used for calculating the C-Index
    with torch.no_grad():
        for i, (x_path, _, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            S = torch.cumprod(1 - logits[2], dim=1) #[B,4]
            risk = -torch.sum(S, dim=1) #[B]
            # logits:[hazard_tumor, hazard_immune, hazard, omic_tumor, vgrid_tumor, omic_immune, vgrid_immune]
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor])
            risk_pred_all = np.concatenate((risk_pred_all, risk.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            # event_all = np.concatenate((event_all, label[:, 10].detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information
        
        # print('risk_pred_all.shape:', risk_pred_all.shape) #236
        # print('event_all.shape:', event_all.shape) #236

        cindex = CIndex_sksurv(all_risk_scores=risk_pred_all, all_censorships=censor_all, all_event_times=survtime_all)

    model.train(training)

    return cindex

# for BaselineModel val
def epochBaselineModelVal(model, dataLoader, args):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        # for i, (x_path, x_omic, label) in enumerate(dataLoader):
            # x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
            # fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
            # output = F.softmax(logits, dim=1)
            # groundTruth = torch.cat((groundTruth, label[:, 5]))
            # activations = torch.cat((activations, output))
        for i, (x_path, _, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            if args.mode == 'path':
                path_vec, logits, _ = model(x_path)  # (BS,2500,1024), x_path x pathology
            elif args.mode == 'omic':
                omic_vec, logits, _ = model(x_omic=x_omic)
            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
            elif args.mode == 'pathomic_fg' or args.mode == 'pathomic_ensemble':
                _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, is_training=model.training)
            elif args.mode == 'mcat':
                logits, hazards, S = model(x_path=x_path, x_omic=x_omic) # return hazards, S, Y_hat, A_raw, results_dict
            elif args.mode == 'cmta':
                # hazards= model(x_path=x_path, x_omic=x_omic) # return hazards, S, Y_hat, A_raw, results_dict
                # hazards, S, cls_token_pathomics_encoder, cls_token_pathomics_decoder, cls_token_genomics_encoder, cls_token_genomics_decoder 
                logits, hazards, S, P, P_hat, G, G_hat = model(x_path=x_path, x_omic=x_omic)

            if args.mode == 'path' or args.mode == 'omic' or args.mode == 'mcat' or args.mode == 'cmta':
                output = F.softmax(logits, dim=1)
            elif args.mode == 'pathomic' or args.mode == 'pathomic_original' or args.mode == 'pathomic_fg' or args.mode == 'pathomic_ensemble':
                output = F.softmax(logits[2], dim=1)
            if args.task_type == "diag2021":
                groundTruth = torch.cat((groundTruth, label[:, 5]))
            elif args.task_type == "grade":
                groundTruth = torch.cat((groundTruth, label[:, 4]))
            elif args.task_type == "subtype":
                groundTruth = torch.cat((groundTruth, label[:, 7]))
            activations = torch.cat((activations, output))
            
        acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return acc, f1, auc, bac, sens, spec, prec

def epochBaselineModelVal_survival(model, dataLoader, args):
    training = model.training
    model.eval()
    
    risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
    with torch.no_grad():
        for i, (x_path, _, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            
            if args.mode == 'path':
                path_vec, logits, _ = model(x_path)  # (BS,2500,1024), x_path x pathology
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1) #[8]
            elif args.mode == 'omic':
                omic_vec, logits, _ = model(x_omic=x_omic)
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1) #[8]
            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
                hazards = torch.sigmoid(logits[2])
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1) #[8]
            elif args.mode == 'pathomic_fg' or args.mode == 'pathomic_ensemble':
                _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, is_training=model.training)
                hazards = torch.sigmoid(logits[2])
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1) #[8]
            elif args.mode == 'mcat':
                logits, hazards, S = model(x_path=x_path, x_omic=x_omic) # return hazards, S, Y_hat, A_raw, results_dict
                risk = -torch.sum(S, dim=1) #[8,4]
            elif args.mode == 'cmta':
                logits, hazards, S, P, P_hat, G, G_hat = model(x_path=x_path, x_omic=x_omic)
                risk = -torch.sum(S, dim=1) #[8]
            
            # logits:[hazard_tumor, hazard_immune, hazard, omic_tumor, vgrid_tumor, omic_immune, vgrid_immune]
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor])
                
            # risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            risk_pred_all = np.concatenate((risk_pred_all, risk.detach().cpu().numpy().reshape(-1)))   # Logging Information
            censor_all = np.concatenate((censor_all, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            # event_all = np.concatenate((event_all, label[:, 10].detach().cpu().numpy().reshape(-1)))
            # print('event_all.shape:', event_all.shape)
            survtime_all = np.concatenate((survtime_all, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information
        
        cindex = CIndex_sksurv(all_risk_scores=risk_pred_all, all_censorships=censor_all, all_event_times=survtime_all)
        # acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return cindex


def ablation_epochVal(model, dataLoader, gene_list_length):
    model.eval()

    with torch.no_grad():
        # Store the importance of each gene feature
        # Ablation study for each gene feature
        difference_acc_list = []
        # for i in range(gene_list_length):
        for i in range(2):
            groundTruth = torch.Tensor().cuda()
            prediction = torch.Tensor().cuda()
            modified_prediction = torch.Tensor().cuda()
            print("Processing {:d}/{:d} gene".format(i+1, gene_list_length))
            for j, (x_path, x_omic, label) in enumerate(dataLoader):
                x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
                
                fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
                output = F.softmax(logits, dim=1)
                groundTruth = torch.cat((groundTruth, label[:, 5]))
                prediction = torch.cat((prediction, output))
                
                ablated_genes = x_omic.clone()
                ablated_genes[:, i] = 0  # Zero out the ith gene feature
                fuse_feat, path_feat, omic_feat, modified_logits, _, _, _ = model(x_path=x_path, x_omic=ablated_genes)
                modified_output = F.softmax(modified_logits, dim=1)
                modified_prediction = torch.cat((modified_prediction, modified_output))

            acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, prediction)
            modified_acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, modified_prediction)
            print('acc, modif_acc:', acc, modified_acc)
            
            difference_acc_list.append(acc - modified_acc) 
            
    return difference_acc_list

def eli5_epochVal(model, x_path, x_omic, label):
    model.eval()
    with torch.no_grad():
        # Store the importance of each gene feature
        # for i in range(gene_list_length):
        x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
                
        fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
        # calculate the accuracy
        pred = F.softmax(logits, dim=1).argmax(dim=1)
        acc = pred.eq(label[:, 5].view_as(pred)).sum().item() / float(label[:, 5].shape[0])
                    
    return acc


def epochTest(model, dataLoader):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            output = model(image)
            if isinstance(output, tuple):
                _, output = output
            output = F.softmax(output, dim=1)
            groundTruth = torch.cat((groundTruth, label))
            activations = torch.cat((activations, output))

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions)
    model.train(training)

    return cm