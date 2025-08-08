# train the encoder
import os
import time
import torch
import wandb
import torch.nn as nn
import torch.distributed as dist
from utils.metrics import epochVal, epochVal_survival, epochDistillVal, epochDistillVal_survival
from utils.loss import BatchLoss, PathBatchLoss, OmicDomainScaleLoss, DistillationLoss
from utils.utils import CIndex_sksurv
import torch.nn.functional as F
from utils.utils import NLLSurvLoss
from models.cmta_utils import define_loss
import pandas as pd
import numpy as np

def trainTeachersModel(model, dataloader, optimizer, scheduler, logger, args):
    # diag2021_loss_func = nn.CrossEntropyLoss() #
    # 636/1828: (1.0, 4.56, 3.21, 2.65) (636, 288, 409, 495)->( 636+676= 1312, 288, 409, 495)
    # diag2021: (1.0, 4.15, 2.93, 2.43)
    # grad: (1.47, 1.51, 1.0)
    # subtype: (1.0, 1.72, 2.43)

    if args.external_eval:
        diag2021_loss_func = nn.CrossEntropyLoss().cuda()
    else:
        diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    survival_loss_func = NLLSurvLoss(alpha=0.15)
    # distill_loss_func = DistillationLoss(temperature=args.temperature)
    batch_sim_loss_func = PathBatchLoss(args.batch_size, args.world_size) 
    omic_domain_loss_func = OmicDomainScaleLoss(args.batch_size, args.world_size) 
            
    start = time.time()
    cur_iters = 0
    model.train()
    train_loader, test_loader = dataloader
    # print('length of trainloader:', len(train_loader)) #36
    # print('length of testloader:', len(test_loader))   #10
    cur_lr = args.lr
    if args.task_type == "survival":
        best_cindex = 0.0
    else:
        best_auc = 0.0
        best_acc = 0.0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if args.task_type == "survival":
            risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(train_loader):
            x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time])
            
            feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)

            # classification loss
            #label[:, 5] is for diagnosis
                
            if args.task_type == "diag2021":
                taskloss_tea10 = diag2021_loss_func(logits_dict['logits_tea10'].float(), label[:, 5])
                taskloss_tea20 = diag2021_loss_func(logits_dict['logits_tea20'].float(), label[:, 5])
                taskloss = taskloss_tea10 + taskloss_tea20
                # print('testing.........')
                # taskloss = taskloss_stu10

            elif args.task_type == "survival":
                taskloss_tea10 = survival_loss_func(hazards=hazards_dict['hazards_tea10'], S=S_dict['S_tea10'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss_tea20 = survival_loss_func(hazards=hazards_dict['hazards_tea20'], S=S_dict['S_tea20'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss = taskloss_tea10 + taskloss_tea20
            elif args.task_type == "grade":
                taskloss_tea10 = grade_loss_func(logits_dict['logits_tea10'].float(), label[:, 4])
                taskloss_tea20 = grade_loss_func(logits_dict['logits_tea20'].float(), label[:, 4])
                taskloss = taskloss_tea10 + taskloss_tea20
            elif args.task_type == "subtype":
                taskloss_tea10 = subtype_loss_func(logits_dict['logits_tea10'].float(), label[:, 7])
                taskloss_tea20 = subtype_loss_func(logits_dict['logits_tea20'].float(), label[:, 7])
                taskloss = taskloss_tea10 + taskloss_tea20
            
            # for distillation loss
            # distillloss = distill_loss_func(logits_dict['logits_tea20'], logits_dict['logits_tea10'])

            # if args.multiscale_attention and epoch>1:
            if args.multiscale_attention:
                # print('attloss_path inputs shape:', att_dict['pathpath_att10'].shape, att_dict['pathpath_att20'].shape)
                # print('attloss_omic1 inputs shape:', att_dict['omic1path_att10'].shape, att_dict['omic1path_att20'].shape)
                # batchloss_histo = torch.sum(batch_sim_loss_func(att_dict['att_stu10'], att_dict['att_stu20']))
                batchloss_mole = torch.sum(omic_domain_loss_func(att_dict['att1_tea10'], att_dict['att1_tea20'],
                                                                    att_dict['att2_tea10'], att_dict['att2_tea20']))
                batchloss = batchloss_mole                
            # loss1 = diag2021_loss_func(logits[0], label[:, 5])
            # loss2 = diag2021_loss_func(logits[1], label[:, 5])
            if args.multiscale_attention:
                loss = taskloss + batchloss
                # print("task&batch loss")
                # loss = batchloss #for batchloss debug!!!
                # print("only batchloss")
            else:
                loss = taskloss  
            
            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            
            # Update parameters based on projected gradients
            if args.gradient_modulate:
                # Align gradients only if they contradict
                hs = args.mmhid
                out1_tea10 = (torch.mm(feature_dict['feature1_tea10'], torch.transpose(model.module.teacher10_net.classifier.weight[:, :hs], 0, 1)) +
                            model.module.teacher10_net.classifier.bias / 2)
                out2_tea10 = (torch.mm(feature_dict['feature2_tea10'], torch.transpose(model.module.teacher10_net.classifier.weight[:, hs:], 0, 1)) +
                            model.module.teacher10_net.classifier.bias / 2)
                out1_tea20 = (torch.mm(feature_dict['feature1_tea20'], torch.transpose(model.module.teacher20_net.classifier.weight[:, :hs], 0, 1)) +
                            model.module.teacher20_net.classifier.bias / 2)
                out2_tea20 = (torch.mm(feature_dict['feature2_tea20'], torch.transpose(model.module.teacher20_net.classifier.weight[:, hs:], 0, 1)) +
                            model.module.teacher20_net.classifier.bias / 2)
                
                # Modulation starts here !
                if args.task_type == "diag2021":
                    score1_tea10 = sum([F.softmax(out1_tea10)[i][label[:, 5][i]] for i in range(out1_tea10.size(0))])
                    score2_tea10 = sum([F.softmax(out2_tea10)[i][label[:, 5][i]] for i in range(out2_tea10.size(0))])
                    score1_tea20 = sum([F.softmax(out1_tea20)[i][label[:, 5][i]] for i in range(out1_tea20.size(0))])
                    score2_tea20 = sum([F.softmax(out2_tea20)[i][label[:, 5][i]] for i in range(out2_tea20.size(0))])
                elif args.task_type == "survival":
                    score1_tea10 = sum([F.softmax(out1_tea10)[i][label[:, 8][i]] for i in range(out1_tea10.size(0))])
                    score2_tea10 = sum([F.softmax(out2_tea10)[i][label[:, 8][i]] for i in range(out2_tea10.size(0))])
                    score1_tea20 = sum([F.softmax(out1_tea20)[i][label[:, 8][i]] for i in range(out1_tea20.size(0))])
                    score2_tea20 = sum([F.softmax(out2_tea20)[i][label[:, 8][i]] for i in range(out2_tea20.size(0))])
                elif args.task_type == "grade":
                    score1_tea10 = sum([F.softmax(out1_tea10)[i][label[:, 4][i]] for i in range(out1_tea10.size(0))])
                    score2_tea10 = sum([F.softmax(out2_tea10)[i][label[:, 4][i]] for i in range(out2_tea10.size(0))])
                    score1_tea20 = sum([F.softmax(out1_tea20)[i][label[:, 4][i]] for i in range(out1_tea20.size(0))])
                    score2_tea20 = sum([F.softmax(out2_tea20)[i][label[:, 4][i]] for i in range(out2_tea20.size(0))])
                elif args.task_type == "subtype":
                    score1_tea10 = sum([F.softmax(out1_tea10)[i][label[:, 7][i]] for i in range(out1_tea10.size(0))])
                    score2_tea10 = sum([F.softmax(out2_tea10)[i][label[:, 7][i]] for i in range(out2_tea10.size(0))])
                    score1_tea20 = sum([F.softmax(out1_tea20)[i][label[:, 7][i]] for i in range(out1_tea20.size(0))])
                    score2_tea20 = sum([F.softmax(out2_tea20)[i][label[:, 7][i]] for i in range(out2_tea20.size(0))])
                
                ratio1_tea10 = score1_tea10 / score2_tea10
                ratio2_tea10 = 1 / ratio1_tea10
                ratio1_tea20 = score1_tea20 / score2_tea20
                ratio2_tea20 = 1 / ratio1_tea20                
                
                # print('ratio_t:', ratio_t)

                if ratio1_tea10 is not None and ratio2_tea10 is not None:
                    i_index=0
                    for grad1_tea10, grad2_tea10 in zip(model.module.teacher10_net.classifier.weight.grad[:, :hs], model.module.teacher10_net.classifier.weight.grad[:, hs:]):
                        if grad1_tea10 is not None and grad2_tea10 is not None:
                            # print('grad_t.shape:', grad_t.shape) #[128]
                            # print('grad_i.shape:', grad_i.shape) #[128]
                            sim = cosine_similarity(grad1_tea10, grad2_tea10)
                            # sim = F.cosine_similarity(grad_t, grad_i)
                            if sim < 0:
                                if ratio1_tea10 < 1:
                                    # Calculate the projection of gradient of classifier_tumor onto the direction perpendicular to gradient of classifier
                                    dot_product = torch.dot(grad1_tea10.flatten(), grad2_tea10.flatten())
                                    proj_scale = dot_product / grad2_tea10.norm()**2
                                    proj_component = proj_scale * grad2_tea10
                                    grad1_tea10 = grad1_tea10 - proj_component
                                    # model.module.classifier.weight.grad[i_index, :hs] = grad_t
                                    perpen = grad1_tea10 - proj_component
                                    unit_perpen = perpen / perpen.norm()
                                    grad1_tea10 = grad1_tea10.norm() * unit_perpen
                                    model.module.teacher10_net.classifier.weight.grad[i_index, :hs] = grad1_tea10
                                elif ratio2_tea10 < 1:
                                    # Calculate the projection of gradient of classifier_tumor onto the direction perpendicular to gradient of classifier
                                    dot_product = torch.dot(grad2_tea10.flatten(), grad1_tea10.flatten())
                                    proj_scale = dot_product / grad1_tea10.norm()**2
                                    proj_component = proj_scale * grad1_tea10
                                    grad2_tea10 = grad2_tea10 - proj_component
                                    # model.module.classifier.weight.grad[i_index, hs:] = grad_i
                                    perpen = grad2_tea10 - proj_component
                                    unit_perpen = perpen / perpen.norm()
                                    grad2_tea10 = grad2_tea10.norm() * unit_perpen   
                                    model.module.teacher10_net.classifier.weight.grad[i_index, hs:] = grad2_tea10
                        i_index = i_index+1

                if ratio1_tea20 is not None and ratio2_tea20 is not None:
                    i_index=0
                    for grad1_tea20, grad2_tea20 in zip(model.module.teacher20_net.classifier.weight.grad[:, :hs], model.module.teacher20_net.classifier.weight.grad[:, hs:]):
                        if grad1_tea20 is not None and grad2_tea20 is not None:
                            # print('grad_t.shape:', grad_t.shape) #[128]
                            # print('grad_i.shape:', grad_i.shape) #[128]
                            sim = cosine_similarity(grad1_tea20, grad2_tea20)
                            # sim = F.cosine_similarity(grad_t, grad_i)
                            if sim < 0:
                                if ratio1_tea20 < 1:
                                    # Calculate the projection of gradient of classifier_tumor onto the direction perpendicular to gradient of classifier
                                    dot_product = torch.dot(grad1_tea20.flatten(), grad2_tea20.flatten())
                                    proj_scale = dot_product / grad2_tea20.norm()**2
                                    proj_component = proj_scale * grad2_tea20
                                    grad1_tea20 = grad1_tea20 - proj_component
                                    # model.module.classifier.weight.grad[i_index, :hs] = grad_t
                                    perpen = grad1_tea20 - proj_component
                                    unit_perpen = perpen / perpen.norm()
                                    grad1_tea20 = grad1_tea20.norm() * unit_perpen
                                    model.module.teacher20_net.classifier.weight.grad[i_index, :hs] = grad1_tea20
                                elif ratio2_tea20 < 1:
                                    # Calculate the projection of gradient of classifier_tumor onto the direction perpendicular to gradient of classifier
                                    dot_product = torch.dot(grad2_tea20.flatten(), grad1_tea20.flatten())
                                    proj_scale = dot_product / grad1_tea20.norm()**2
                                    proj_component = proj_scale * grad1_tea20
                                    grad2_tea20 = grad2_tea20 - proj_component
                                    # model.module.classifier.weight.grad[i_index, hs:] = grad_i
                                    perpen = grad2_tea20 - proj_component
                                    unit_perpen = perpen / perpen.norm()
                                    grad2_tea20 = grad2_tea20.norm() * unit_perpen   
                                    model.module.teacher20_net.classifier.weight.grad[i_index, hs:] = grad2_tea20
                        i_index = i_index+1

            # if dist.is_available() and dist.is_initialized():
            #     loss = loss.data.clone()
            #     dist.all_reduce(loss.div_(dist.get_world_size()))

            # 2. Synchronize gradients across all processes 同步梯度
            if dist.is_available() and dist.is_initialized():
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad.data,op=dist.ReduceOp.SUM) #Sum gradients
                        p.grad.data /= dist.get_world_size()

            optimizer.step()
            scheduler.step()

            cur_iters += 1
            if args.rank == 0: 
                if cur_iters % 10 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    # evaluate on test and val set
                    if args.task_type == "survival":
                        test_cindex_dict = epochDistillVal_survival(model, test_loader, args, model_type='teacher')
                        if logger is not None and args.multiscale_attention:
                            logger.log({'training': {'total loss': loss.item(),
                                                    'taskloss': taskloss.item(),
                                                    # 'distillloss': distillloss.item(),
                                                    'batchloss': batchloss.item()}})
                            logger.log({'test': {'cindex_teas': test_cindex_dict['cindex_teas'],
                                                 'cindex_tea10': test_cindex_dict['cindex_tea10'],
                                                 'cindex_tea20': test_cindex_dict['cindex_tea20'],}})
                        elif logger is not None:
                            logger.log({'training': {'total loss': loss.item()}})
                            logger.log({'test': {'cindex_teas': test_cindex_dict['cindex_teas'],
                                                 'cindex_tea10': test_cindex_dict['cindex_tea10'],
                                                 'cindex_tea20': test_cindex_dict['cindex_tea20'],}})
                    else:
                        test_acc_dict, test_f1_dict, test_auc_dict, test_bac_dict, test_sens_dict, test_spec_dict, test_prec_dict = epochDistillVal(model, test_loader, args, model_type='teacher')
                        if logger is not None and args.multiscale_attention:
                            current_lr = optimizer.param_groups[0]['lr']
                            logger.log({"learning_rate": current_lr})
                            logger.log({'training': {'total loss': loss.item(),
                                                     'taskloss': taskloss.item(),
                                                    #  'distillloss': distillloss.item(),
                                                     'batchloss': batchloss.item()}})
                            logger.log({'test_teas': {'Accuracy': test_acc_dict['acc_teas'],
                                                    'F1 score': test_f1_dict['f1_teas'],
                                                    'AUC': test_auc_dict['auc_teas'],
                                                    'Balanced Accuracy': test_bac_dict['bac_teas'],
                                                    'Sensitivity': test_sens_dict['sens_teas'],
                                                    'Specificity': test_spec_dict['spec_teas'],
                                                    'Precision': test_prec_dict['prec_teas']},
                                        'test_tea10': {'Accuracy': test_acc_dict['acc_tea10'],
                                                    'F1 score': test_f1_dict['f1_tea10'],
                                                    'AUC': test_auc_dict['auc_tea10'],
                                                    'Balanced Accuracy': test_bac_dict['bac_tea10'],
                                                    'Sensitivity': test_sens_dict['sens_tea10'],
                                                    'Specificity': test_spec_dict['spec_tea10'],
                                                    'Precision': test_prec_dict['prec_tea10']},
                                        'test_tea20': {'Accuracy': test_acc_dict['acc_tea20'],
                                                    'F1 score': test_f1_dict['f1_tea20'],
                                                    'AUC': test_auc_dict['auc_tea20'],
                                                    'Balanced Accuracy': test_bac_dict['bac_tea20'],
                                                    'Sensitivity': test_sens_dict['sens_tea20'],
                                                    'Specificity': test_spec_dict['spec_tea20'],
                                                    'Precision': test_prec_dict['prec_tea20']},
                                                   })
                        elif logger is not None:
                            current_lr = optimizer.param_groups[0]['lr']
                            logger.log({"learning_rate": current_lr})
                            logger.log({'training': {'total loss': loss.item()}})
                            logger.log({'test_teas': {'Accuracy': test_acc_dict['acc_teas'],
                                                    'F1 score': test_f1_dict['f1_teas'],
                                                    'AUC': test_auc_dict['auc_teas'],
                                                    'Balanced Accuracy': test_bac_dict['bac_teas'],
                                                    'Sensitivity': test_sens_dict['sens_teas'],
                                                    'Specificity': test_spec_dict['spec_teas'],
                                                    'Precision': test_prec_dict['prec_teas']},
                                        'test_tea10': {'Accuracy': test_acc_dict['acc_tea10'],
                                                    'F1 score': test_f1_dict['f1_tea10'],
                                                    'AUC': test_auc_dict['auc_tea10'],
                                                    'Balanced Accuracy': test_bac_dict['bac_tea10'],
                                                    'Sensitivity': test_sens_dict['sens_tea10'],
                                                    'Specificity': test_spec_dict['spec_tea10'],
                                                    'Precision': test_prec_dict['prec_tea10']},
                                        'test_tea20': {'Accuracy': test_acc_dict['acc_tea20'],
                                                    'F1 score': test_f1_dict['f1_tea20'],
                                                    'AUC': test_auc_dict['auc_tea20'],
                                                    'Balanced Accuracy': test_bac_dict['bac_tea20'],
                                                    'Sensitivity': test_sens_dict['sens_tea20'],
                                                    'Specificity': test_spec_dict['spec_tea20'],
                                                    'Precision': test_prec_dict['prec_tea20']},
                                                   })
                    
                    if not args.multiscale_attention:
                        print('\rFold: [%2d/%2d] Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f || LossTask: %.4f' % (
                            args.cur_fold, args.kfold, epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item(), taskloss.item()), end='', flush=True)
                    elif args.multiscale_attention:
                        print('\rFold: [%2d/%2d] Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f || LossTask: %.4f || LossBatch: %.4f' % (
                            args.cur_fold, args.kfold, epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item(), taskloss.item(), batchloss.item()), end='', flush=True)  
        # scheduler.step()
        
        # method2: save best model

    if args.rank == 0:
        if args.task_type == "survival":
            test_cindex_dict = epochDistillVal_survival(model, test_loader, args, model_type='teacher')
            saveModelPath = os.path.join(args.checkpoints, 'fold_{:d}_epoch_{:d}_cindex_{:f}_.pth'.format(
                args.cur_fold+1, epoch + 1, test_cindex_dict['cindex_teas'])) 
            if dist.is_available() and dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()        
            torch.save(state_dict, saveModelPath)    
        else: 
            test_acc_dict, test_f1_dict, test_auc_dict, test_bac_dict, test_sens_dict, test_spec_dict, test_prec_dict = epochDistillVal(model, test_loader, args, model_type='teacher')
            saveModelPath = os.path.join(args.checkpoints, 'fold_{:d}_epoch_{:d}_AUC_{:f}_ACC_{:f}_Sens_{:f}_Spec_{:f}_F1_{:f}_.pth'.format(
                args.cur_fold+1,epoch + 1, test_auc_dict['auc_teas'], test_acc_dict['acc_teas'], test_sens_dict['sens_teas'], test_spec_dict['spec_teas'], test_f1_dict['f1_teas']))
            if dist.is_available() and dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()        
            torch.save(state_dict, saveModelPath)

def trainStudentsModel(model, dataloader, optimizer, scheduler, logger, args):
    # diag2021_loss_func = nn.CrossEntropyLoss(label_smoothing=0.1) #
    # 636/1828: (1.0, 4.56, 3.21, 2.65) (636, 288, 409, 495)->( 636+676= 1312, 288, 409, 495)
    # diag2021: (1.0, 4.15, 2.93, 2.43)
    # grad: (1.47, 1.51, 1.0)
    # subtype: (1.0, 1.72, 2.43)
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    survival_loss_func = NLLSurvLoss(alpha=0.15)
    # distill_loss_func = DistillationLoss(temperature=args.temperature)
    batch_sim_loss_func = PathBatchLoss(args.batch_size, args.world_size) 
    omic_domain_loss_func = OmicDomainScaleLoss(args.batch_size, args.world_size) 
            
    start = time.time()
    cur_iters = 0
    model.train()
    train_loader, test_loader = dataloader
    # print('length of trainloader:', len(train_loader)) #36
    # print('length of testloader:', len(test_loader))   #10
    cur_lr = args.lr
    if args.task_type == "survival":
        best_cindex = 0.0
    else:
        best_auc = 0.0
        best_acc = 0.0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if args.task_type == "survival":
            risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(train_loader):
            x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time])
            
            feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20)

            # classification loss
            #label[:, 5] is for diagnosis
                
            if args.task_type == "diag2021":
                taskloss_stu10 = diag2021_loss_func(logits_dict['logits_stu10'].float(), label[:, 5])
                taskloss_stu20 = diag2021_loss_func(logits_dict['logits_stu20'].float(), label[:, 5])
                taskloss = taskloss_stu10 + taskloss_stu20
                # print('testing.........')
                # taskloss = taskloss_stu10

            elif args.task_type == "survival":
                taskloss_stu10 = survival_loss_func(hazards=hazards_dict['hazards_stu10'], S=S_dict['S_stu10'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss_stu20 = survival_loss_func(hazards=hazards_dict['hazards_stu20'], S=S_dict['S_stu20'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss = taskloss_stu10 + taskloss_stu20
            elif args.task_type == "grade":
                taskloss_stu10 = grade_loss_func(logits_dict['logits_stu10'].float(), label[:, 4])
                taskloss_stu20 = grade_loss_func(logits_dict['logits_stu20'].float(), label[:, 4])
                taskloss = taskloss_stu10 + taskloss_stu20
            elif args.task_type == "subtype":
                taskloss_stu10 = subtype_loss_func(logits_dict['logits_stu10'].float(), label[:, 7])
                taskloss_stu20 = subtype_loss_func(logits_dict['logits_stu20'].float(), label[:, 7])
                taskloss = taskloss_stu10 + taskloss_stu20
            
            # # for distillation loss
            # distillloss = distill_loss_func(logits_dict['logits_stu20'], logits_dict['logits_stu10'])

            # if args.multiscale_attention and epoch>1:
            if args.multiscale_attention:
                # print('attloss_path inputs shape:', att_dict['pathpath_att10'].shape, att_dict['pathpath_att20'].shape)
                # print('attloss_omic1 inputs shape:', att_dict['omic1path_att10'].shape, att_dict['omic1path_att20'].shape)
                batchloss_histo = torch.sum(batch_sim_loss_func(att_dict['att_stu10'], att_dict['att_stu20']))
                # batch_sim_loss = 0.5*batch_sim_loss_path + 0.5*batch_sim_loss_omic  
                batchloss = 1000*batchloss_histo             
            # loss1 = diag2021_loss_func(logits[0], label[:, 5])
            # loss2 = diag2021_loss_func(logits[1], label[:, 5])
            if args.multiscale_attention:
                loss = taskloss + batchloss
            else:
                loss = taskloss  
            
            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            
            # 2. Synchronize gradients across all processes 同步梯度
            if dist.is_available() and dist.is_initialized():
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad.data,op=dist.ReduceOp.SUM) #Sum gradients
                        p.grad.data /= dist.get_world_size()

            optimizer.step()
            scheduler.step()

            cur_iters += 1
            if args.rank == 0: 
                if cur_iters % 10 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    # evaluate on test and val set
                    if args.task_type == "survival":
                        test_cindex_dict = epochDistillVal_survival(model, test_loader, args, model_type='student')
                        if logger is not None and args.multiscale_attention:
                            logger.log({'training': {'total loss': loss.item(),
                                                    'taskloss': taskloss.item(),
                                                    # 'distillloss': distillloss.item(),
                                                    'batchloss': batchloss.item()}})
                            logger.log({'test': {'cindex_stu10': test_cindex_dict['cindex_stu10'],
                                                 'cindex_stu20': test_cindex_dict['cindex_stu20'],
                                                 'cindex_stus': test_cindex_dict['cindex_stus'],}})
                        elif logger is not None:
                            logger.log({'training': {'total loss': loss.item()}})
                            logger.log({'test': {'cindex_stu10': test_cindex_dict['cindex_stu10'],
                                                 'cindex_stu20': test_cindex_dict['cindex_stu20'],
                                                 'cindex_stus': test_cindex_dict['cindex_stus'],}})
                    else:
                        test_acc_dict, test_f1_dict, test_auc_dict, test_bac_dict, test_sens_dict, test_spec_dict, test_prec_dict = epochDistillVal(model, test_loader, args, model_type='student')
                        if logger is not None and args.multiscale_attention:
                            current_lr = optimizer.param_groups[0]['lr']
                            logger.log({"learning_rate": current_lr})
                            logger.log({'training': {'total loss': loss.item(),
                                                     'taskloss': taskloss.item(),
                                                    #  'distillloss': distillloss.item(),
                                                     'batchloss': batchloss.item()}})
                            logger.log({'test_stu10': {'Accuracy': test_acc_dict['acc_stu10'],
                                                    'F1 score': test_f1_dict['f1_stu10'],
                                                    'AUC': test_auc_dict['auc_stu10'],
                                                    'Balanced Accuracy': test_bac_dict['bac_stu10'],
                                                    'Sensitivity': test_sens_dict['sens_stu10'],
                                                    'Specificity': test_spec_dict['spec_stu10'],
                                                    'Precision': test_prec_dict['prec_stu10']},
                                        'test_stu20': {'Accuracy': test_acc_dict['acc_stu20'],
                                                    'F1 score': test_f1_dict['f1_stu20'],
                                                    'AUC': test_auc_dict['auc_stu20'],
                                                    'Balanced Accuracy': test_bac_dict['bac_stu20'],
                                                    'Sensitivity': test_sens_dict['sens_stu20'],
                                                    'Specificity': test_spec_dict['spec_stu20'],
                                                    'Precision': test_prec_dict['prec_stu20']},
                                        'test_stus': {'Accuracy': test_acc_dict['acc_stus'],
                                                    'F1 score': test_f1_dict['f1_stus'],
                                                    'AUC': test_auc_dict['auc_stus'],
                                                    'Balanced Accuracy': test_bac_dict['bac_stus'],
                                                    'Sensitivity': test_sens_dict['sens_stus'],
                                                    'Specificity': test_spec_dict['spec_stus'],
                                                    'Precision': test_prec_dict['prec_stus']},
                                                   })
                        elif logger is not None:
                            current_lr = optimizer.param_groups[0]['lr']
                            logger.log({"learning_rate": current_lr})
                            logger.log({'training': {'total loss': loss.item()}})
                            logger.log({'test_stu10': {'Accuracy': test_acc_dict['acc_stu10'],
                                                    'F1 score': test_f1_dict['f1_stu10'],
                                                    'AUC': test_auc_dict['auc_stu10'],
                                                    'Balanced Accuracy': test_bac_dict['bac_stu10'],
                                                    'Sensitivity': test_sens_dict['sens_stu10'],
                                                    'Specificity': test_spec_dict['spec_stu10'],
                                                    'Precision': test_prec_dict['prec_stu10']},
                                        'test_stu20': {'Accuracy': test_acc_dict['acc_stu20'],
                                                    'F1 score': test_f1_dict['f1_stu20'],
                                                    'AUC': test_auc_dict['auc_stu20'],
                                                    'Balanced Accuracy': test_bac_dict['bac_stu20'],
                                                    'Sensitivity': test_sens_dict['sens_stu20'],
                                                    'Specificity': test_spec_dict['spec_stu20'],
                                                    'Precision': test_prec_dict['prec_stu20']},
                                        'test_stus': {'Accuracy': test_acc_dict['acc_stus'],
                                                    'F1 score': test_f1_dict['f1_stus'],
                                                    'AUC': test_auc_dict['auc_stus'],
                                                    'Balanced Accuracy': test_bac_dict['bac_stus'],
                                                    'Sensitivity': test_sens_dict['sens_stus'],
                                                    'Specificity': test_spec_dict['spec_stus'],
                                                    'Precision': test_prec_dict['prec_stus']},
                                                   })
                    
                    if not args.multiscale_attention:
                        print('\rFold: [%2d/%2d] Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f || LossTask: %.4f' % (
                            args.cur_fold, args.kfold, epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item(), taskloss.item()), end='', flush=True)
                    elif args.multiscale_attention:
                        print('\rFold: [%2d/%2d] Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f || LossTask: %.4f || LossBatch: %.4f' % (
                            args.cur_fold, args.kfold, epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item(), taskloss.item(), batchloss.item()), end='', flush=True)  
        # scheduler.step()
        
        # method2: save best model

    if args.rank == 0:
        if args.task_type == "survival":
            test_cindex_dict = epochDistillVal_survival(model, test_loader, args, model_type='student')
            saveModelPath = os.path.join(args.checkpoints, 'fold_{:d}_epoch_{:d}_cindex_{:f}_.pth'.format(
                args.cur_fold+1, epoch + 1, test_cindex_dict['cindex_stus'])) 
            if dist.is_available() and dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()        
            torch.save(state_dict, saveModelPath)    
        else: 
            test_acc_dict, test_f1_dict, test_auc_dict, test_bac_dict, test_sens_dict, test_spec_dict, test_prec_dict = epochDistillVal(model, test_loader, args, model_type='student')

            saveModelPath = os.path.join(args.checkpoints, 'fold_{:d}_epoch_{:d}_AUC_{:f}_ACC_{:f}_Sens_{:f}_Spec_{:f}_F1_{:f}_.pth'.format(
                args.cur_fold+1, epoch + 1, test_auc_dict['auc_stus'], test_acc_dict['acc_stus'], test_sens_dict['sens_stus'], test_spec_dict['spec_stus'], test_f1_dict['f1_stus']))
            if dist.is_available() and dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()        
            torch.save(state_dict, saveModelPath)

def trainDistillation(student_model, teacher_model, dataloader, optimizer, scheduler, logger, args):
    student_model.train()
    teacher_model.eval()
    # diag2021_loss_func = nn.CrossEntropyLoss() #
    # 636/1828: (1.0, 4.56, 3.21, 2.65) (636, 288, 409, 495)->( 636+676= 1312, 288, 409, 495)
    # diag2021: (1.0, 4.15, 2.93, 2.43)
    # grad: (1.47, 1.51, 1.0)
    # subtype: (1.0, 1.72, 2.43)
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    survival_loss_func = NLLSurvLoss(alpha=0.15)
    distill_loss_func = DistillationLoss(temperature=args.temperature)
    batch_sim_loss_func = PathBatchLoss(args.batch_size, args.world_size) 
            
    start = time.time()
    cur_iters = 0
    # model.train()
    train_loader, test_loader = dataloader
    # print('length of trainloader:', len(train_loader)) #36
    # print('length of testloader:', len(test_loader))   #10
    cur_lr = args.lr
    if args.task_type == "survival":
        best_cindex = 0.0
    else:
        best_auc = 0.0
        best_acc = 0.0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if args.task_type == "survival":
            risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(train_loader):
            x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time])
            
            # Get soft targets from teacher
            with torch.no_grad():
                feature_dict_tea, att_dict_tea, logits_dict_tea, hazards_dict_tea, S_dict_tea, risk_dict_tea = teacher_model(x_path10=x_path10, x_path20=x_path20, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
                feature_tea10 = torch.cat((feature_dict_tea['feature1_tea10'], feature_dict_tea['feature2_tea10']), dim=-1)
                feature_tea20 = torch.cat((feature_dict_tea['feature1_tea20'], feature_dict_tea['feature2_tea20']), dim=-1)
            # Get student outputs
            feature_dict_stu, att_dict_stu, logits_dict_stu, hazards_dict_stu, S_dict_stu, risk_dict_stu = student_model(x_path10=x_path10, x_path20=x_path20)
            feature_stu10 = feature_dict_stu['feature_stu10']
            feature_stu20 = feature_dict_stu['feature_stu20']

            # classification loss
            #label[:, 5] is for diagnosis
                
            if args.task_type == "diag2021":
                taskloss_stu10 = diag2021_loss_func(logits_dict_stu['logits_stu10'].float(), label[:, 5])
                taskloss_stu20 = diag2021_loss_func(logits_dict_stu['logits_stu20'].float(), label[:, 5])
                taskloss = taskloss_stu10 + taskloss_stu20
                # print('testing.........')
                # taskloss = taskloss_stu10

            elif args.task_type == "survival":
                taskloss_stu10 = survival_loss_func(hazards=hazards_dict_stu['hazards_stu10'], S=S_dict_stu['S_stu10'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss_stu20 = survival_loss_func(hazards=hazards_dict_stu['hazards_stu20'], S=S_dict_stu['S_stu20'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss = taskloss_stu10 + taskloss_stu20
            elif args.task_type == "grade":
                taskloss_stu10 = grade_loss_func(logits_dict_stu['logits_stu10'].float(), label[:, 4])
                taskloss_stu20 = grade_loss_func(logits_dict_stu['logits_stu20'].float(), label[:, 4])
                taskloss = taskloss_stu10 + taskloss_stu20
            elif args.task_type == "subtype":
                taskloss_stu10 = subtype_loss_func(logits_dict_stu['logits_stu10'].float(), label[:, 7])
                taskloss_stu20 = subtype_loss_func(logits_dict_stu['logits_stu20'].float(), label[:, 7])
                taskloss = taskloss_stu10 + taskloss_stu20
            
            # for distillation loss
            if args.distill_logits:
                distillloss_logits_10 = distill_loss_func(logits_dict_stu['logits_stu10'], logits_dict_tea['logits_tea10'])
                distillloss_logits_20 = distill_loss_func(logits_dict_stu['logits_stu20'], logits_dict_tea['logits_tea20'])
                distillloss_logits = distillloss_logits_10 + distillloss_logits_20
            if args.distill_feature:
                distillloss_feature_10 = F.mse_loss(feature_stu10, feature_tea10)
                distillloss_feature_20 = F.mse_loss(feature_stu20, feature_tea20)
                distillloss_feature = distillloss_feature_10 + distillloss_feature_20
            # if args.multiscale_attention and epoch>1:
            if args.multiscale_attention:
                # print('attloss_path inputs shape:', att_dict['pathpath_att10'].shape, att_dict['pathpath_att20'].shape)
                # print('attloss_omic1 inputs shape:', att_dict['omic1path_att10'].shape, att_dict['omic1path_att20'].shape)
                batchloss_histo = torch.sum(batch_sim_loss_func(att_dict_stu['att_stu10'], att_dict_stu['att_stu20']))
                # batch_sim_loss = 0.5*batch_sim_loss_path + 0.5*batch_sim_loss_omic  
                batchloss = batchloss_histo           
            # loss1 = diag2021_loss_func(logits[0], label[:, 5])
            # loss2 = diag2021_loss_func(logits[1], label[:, 5])
            if args.distill_logits and args.distill_feature:
                loss = taskloss + 0.01*distillloss_logits + 0.01*distillloss_feature
            elif args.distill_logits and not args.distill_feature:
                loss = taskloss + distillloss_logits
            elif not args.distill_logits and args.distill_feature:
                loss = taskloss + distillloss_feature
            else:
                loss = taskloss  
            
            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()

            # 2. Synchronize gradients across all processes 同步梯度
            if dist.is_available() and dist.is_initialized():
                for p in student_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad.data,op=dist.ReduceOp.SUM) #Sum gradients
                        p.grad.data /= dist.get_world_size()

            optimizer.step()
            scheduler.step()

            cur_iters += 1
            if args.rank == 0: 
                if cur_iters % 10 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    # evaluate on test and val set
                    if args.task_type == "survival":
                        test_cindex_dict_stu = epochDistillVal_survival(student_model, test_loader, args, model_type='student')
                        if logger is not None and args.multiscale_attention:
                            logger.log({'training': {'total loss': loss.item(),
                                                    'taskloss': taskloss.item(),
                                                    'distillloss_logits': distillloss_logits.item(),
                                                    'distillloss_feature': distillloss_feature.item(),
                                                    'batchloss': batchloss.item()}})
                            logger.log({'test': {'cindex_stu10': test_cindex_dict_stu['cindex_stu10'],
                                                 'cindex_stu20': test_cindex_dict_stu['cindex_stu20'],
                                                 'cindex_stus': test_cindex_dict_stu['cindex_stus'],}})
                        elif logger is not None:
                            logger.log({'training': {'total loss': loss.item(),
                                                     'taskloss': taskloss.item(),
                                                     'distillloss_logits': distillloss_logits.item(),
                                                     'distillloss_feature': distillloss_feature.item(),}})
                            logger.log({'test': {'cindex_stu10': test_cindex_dict_stu['cindex_stu10'],
                                                 'cindex_stu20': test_cindex_dict_stu['cindex_stu20'],
                                                 'cindex_stus': test_cindex_dict_stu['cindex_stus'],}})
                    else:
                        test_acc_dict_stu, test_f1_dict_stu, test_auc_dict_stu, test_bac_dict_stu, test_sens_dict_stu, test_spec_dict_stu, test_prec_dict_stu = epochDistillVal(student_model, test_loader, args, model_type='student')
                        if logger is not None and args.multiscale_attention:
                            logger.log({'training': {'total loss': loss.item(),
                                                     'taskloss': taskloss.item(),
                                                     'distillloss_logits': distillloss_logits.item(),
                                                     'distillloss_feature': distillloss_feature.item(),
                                                     'batchloss': batchloss.item()}})
                            logger.log({'test_stu10': {'Accuracy': test_acc_dict_stu['acc_stu10'],
                                                    'F1 score': test_f1_dict_stu['f1_stu10'],
                                                    'AUC': test_auc_dict_stu['auc_stu10'],
                                                    'Balanced Accuracy': test_bac_dict_stu['bac_stu10'],
                                                    'Sensitivity': test_sens_dict_stu['sens_stu10'],
                                                    'Specificity': test_spec_dict_stu['spec_stu10'],
                                                    'Precision': test_prec_dict_stu['prec_stu10']},
                                        'test_stu20': {'Accuracy': test_acc_dict_stu['acc_stu20'],
                                                    'F1 score': test_f1_dict_stu['f1_stu20'],
                                                    'AUC': test_auc_dict_stu['auc_stu20'],
                                                    'Balanced Accuracy': test_bac_dict_stu['bac_stu20'],
                                                    'Sensitivity': test_sens_dict_stu['sens_stu20'],
                                                    'Specificity': test_spec_dict_stu['spec_stu20'],
                                                    'Precision': test_prec_dict_stu['prec_stu20']},
                                        'test_stus': {'Accuracy': test_acc_dict_stu['acc_stus'],
                                                    'F1 score': test_f1_dict_stu['f1_stus'],
                                                    'AUC': test_auc_dict_stu['auc_stus'],
                                                    'Balanced Accuracy': test_bac_dict_stu['bac_stus'],
                                                    'Sensitivity': test_sens_dict_stu['sens_stus'],
                                                    'Specificity': test_spec_dict_stu['spec_stus'],
                                                    'Precision': test_prec_dict_stu['prec_stus']},
                                                   })
                        elif logger is not None:
                            # Log the learning rate to WandB
                            current_lr = optimizer.param_groups[0]['lr']
                            logger.log({"learning_rate": current_lr})
                            logger.log({'training': {'total loss': loss.item(),
                                                     'taskloss': taskloss.item(),
                                                     'distillloss_logits': distillloss_logits.item(),
                                                     'distillloss_feature': distillloss_feature.item(),}})
                            logger.log({'test_stu10': {'Accuracy': test_acc_dict_stu['acc_stu10'],
                                                    'F1 score': test_f1_dict_stu['f1_stu10'],
                                                    'AUC': test_auc_dict_stu['auc_stu10'],
                                                    'Balanced Accuracy': test_bac_dict_stu['bac_stu10'],
                                                    'Sensitivity': test_sens_dict_stu['sens_stu10'],
                                                    'Specificity': test_spec_dict_stu['spec_stu10'],
                                                    'Precision': test_prec_dict_stu['prec_stu10']},
                                        'test_stu20': {'Accuracy': test_acc_dict_stu['acc_stu20'],
                                                    'F1 score': test_f1_dict_stu['f1_stu20'],
                                                    'AUC': test_auc_dict_stu['auc_stu20'],
                                                    'Balanced Accuracy': test_bac_dict_stu['bac_stu20'],
                                                    'Sensitivity': test_sens_dict_stu['sens_stu20'],
                                                    'Specificity': test_spec_dict_stu['spec_stu20'],
                                                    'Precision': test_prec_dict_stu['prec_stu20']},
                                        'test_stus': {'Accuracy': test_acc_dict_stu['acc_stus'],
                                                    'F1 score': test_f1_dict_stu['f1_stus'],
                                                    'AUC': test_auc_dict_stu['auc_stus'],
                                                    'Balanced Accuracy': test_bac_dict_stu['bac_stus'],
                                                    'Sensitivity': test_sens_dict_stu['sens_stus'],
                                                    'Specificity': test_spec_dict_stu['spec_stus'],
                                                    'Precision': test_prec_dict_stu['prec_stus']},
                                                   })
                    
                    if not args.multiscale_attention:
                        print('\rFold: [%2d/%2d] Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f || LossTask: %.4f || LossDistill_logits: %.4f|| LossDistill_feature: %.4f' % (
                            args.cur_fold, args.kfold, epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item(), taskloss.item(), distillloss_logits.item(), distillloss_feature.item()), end='', flush=True)
                    elif args.multiscale_attention:
                        print('\rFold: [%2d/%2d] Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f || LossTask: %.4f || LossDistill_logits: %.4f || LossDistill_feature: %.4f|| LossBatch: %.4f' % (
                            args.cur_fold, args.kfold, epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item(), taskloss.item(), distillloss_logits.item(), distillloss_feature.item(), batchloss.item()), end='', flush=True)  
        # scheduler.step() #updatae in each epoch
        
        # method2: save best model

    if args.rank == 0:
        if args.task_type == "survival":
            test_cindex_dict_stu = epochDistillVal_survival(student_model, test_loader, args, model_type='student')
            saveModelPath = os.path.join(args.checkpoints, 'fold_{:d}_epoch_{:d}_cindex_{:f}_.pth'.format(
                args.cur_fold+1, epoch + 1, test_cindex_dict_stu['cindex_stus'])) 
            if dist.is_available() and dist.is_initialized():
                state_dict = student_model.module.state_dict()
            else:
                state_dict = student_model.state_dict()        
            torch.save(state_dict, saveModelPath)    
        else: 
            test_acc_dict_stu, test_f1_dict_stu, test_auc_dict_stu, test_bac_dict_stu, test_sens_dict_stu, test_spec_dict_stu, test_prec_dict_stu = epochDistillVal(student_model, test_loader, args, model_type='student')
            saveModelPath = os.path.join(args.checkpoints, 'fold_{:d}_epoch_{:d}_AUC_{:f}_ACC_{:f}_Sens_{:f}_Spec_{:f}_F1_{:f}_.pth'.format(
                args.cur_fold+1, epoch + 1, test_auc_dict_stu['auc_stus'], test_acc_dict_stu['acc_stus'], test_sens_dict_stu['sens_stus'], test_spec_dict_stu['spec_stus'], test_f1_dict_stu['f1_stus']))
            if dist.is_available() and dist.is_initialized():
                state_dict = student_model.module.state_dict()
            else:
                state_dict = student_model.state_dict()        
            torch.save(state_dict, saveModelPath)
 
# Function to calculate cosine similarity
def cosine_similarity(grad1, grad2):
    sim = torch.dot(grad1.flatten(), grad2.flatten()) / (grad1.norm() * grad2.norm())
    return sim

def trainDeformPathomicModel(model, dataloader, optimizer, scheduler, logger, args):
    # diag2021_loss_func = nn.CrossEntropyLoss() #
    # 636/1828: (1.0, 4.56, 3.21, 2.65) (636, 288, 409, 495)->( 636+676= 1312, 288, 409, 495)
    # diag2021: (1.0, 4.15, 2.93, 2.43)
    # grad: (1.47, 1.51, 1.0)
    # subtype: (1.0, 1.72, 2.43)
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    # survival_loss_func = NLLSurvLoss(alpha=0.0)
    nll_loss_func = NLLSurvLoss(alpha=0.15)
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)  
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh() 
            
    start = time.time()
    cur_iters = 0
    model.train()
    train_loader, test_loader = dataloader
    # print('length of trainloader:', len(train_loader)) #36
    # print('length of testloader:', len(test_loader))   #10
    cur_lr = args.lr
    if args.task_type == "survival":
        best_cindex = 0.0
    else:
        best_auc = 0.0
        best_acc = 0.0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if args.task_type == "survival":
            risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        for i, (x_path, _, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(train_loader):
        # for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(train_loader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor])
            
            # features, path_vec, omic_vec, logits, pred, pred_path, pred_omic, fuse_grads, path_grads, omic_grads
            # fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            # print('logits[2].shape:', logits[2].shape)#[8,4]
            S = torch.cumprod(1 - logits[2], dim=1)
            # print('S.shape:', S.shape) #[8,4]

            # classification loss
            # loss = diag2021_loss_func(logits[2], label[:, 5])
            # print('label.shape:', label.shape) #torch.Size([2, 7], 
            # logits[2] for tumor+immune diagnosis; label[:, 5] is for diagnosis
                
            if args.task_type == "diag2021":
                loss3 = diag2021_loss_func(logits[2], label[:, 5])
            elif args.task_type == "survival":
                hazard_pred = logits[2]
                loss_nll = nll_loss_func(hazards=hazard_pred, S=S, Y=label[:,8], c=label[:,9], alpha=0)
                loss3 = loss_nll

            elif args.task_type == "grade":
                loss3 = grade_loss_func(logits[2], label[:, 4])
            elif args.task_type == "subtype":
                loss3 = subtype_loss_func(logits[2], label[:, 7])
            if args.return_vgrid:
                batch_sim_loss_tumor = torch.sum(batch_sim_loss_func(logits[3], logits[4]))
                batch_sim_loss_immune = torch.sum(batch_sim_loss_func(logits[5], logits[6]))
                batch_sim_loss = 0.5*batch_sim_loss_tumor + 0.5*batch_sim_loss_immune
                loss = loss3+batch_sim_loss
            # loss1 = diag2021_loss_func(logits[0], label[:, 5])
            # loss2 = diag2021_loss_func(logits[1], label[:, 5])
            else:
                loss = loss3       
            
            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            
            if args.gradient_modulate:
                # Align gradients only if they contradict
                hs = args.mmhid
                out_t = (torch.mm(pathomic_feat_tumor, torch.transpose(model.module.classifier.weight[:, :hs], 0, 1)) +
                            model.module.classifier.bias / 2)
                out_i = (torch.mm(pathomic_feat_immune, torch.transpose(model.module.classifier.weight[:, hs:], 0, 1)) +
                            model.module.classifier.bias / 2)
                
                if args.task_type == "diag2021":
                    loss_t = diag2021_loss_func(out_t, label[:, 5])
                    loss_i = diag2021_loss_func(out_i, label[:, 5])
                elif args.task_type == "survival":
                    hazard_pred_t = torch.sigmoid(out_t)
                    hazard_pred_i = torch.sigmoid(out_i)
                    S_t = torch.cumprod(1 - hazard_pred_t, dim=1)
                    S_i = torch.cumprod(1 - hazard_pred_i, dim=1)

                    loss_nll_t = nll_loss_func(hazards=hazard_pred_t, S=S_t, Y=label[:,8], c=label[:,9], alpha=0)
                    loss_nll_i = nll_loss_func(hazards=hazard_pred_i, S=S_i, Y=label[:,8], c=label[:,9], alpha=0)
                    
                    loss_t = loss_nll_t
                    loss_i = loss_nll_i

                elif args.task_type == "grade":
                    loss_t = grade_loss_func(out_t, label[:, 4])
                    loss_i = grade_loss_func(out_i, label[:, 4])
                elif args.task_type == "subtype":
                    loss_t = subtype_loss_func(out_t, label[:, 7])
                    loss_i = subtype_loss_func(out_i, label[:, 7])
                    
                # Modulation starts here !
                if args.task_type == "diag2021":
                    score_t = sum([F.softmax(out_t)[i][label[:, 5][i]] for i in range(out_t.size(0))])
                    score_i = sum([F.softmax(out_i)[i][label[:, 5][i]] for i in range(out_i.size(0))])
                elif args.task_type == "survival":
                    # # use cindex values 
                    # risk_t = -torch.sum(S_t, dim=1) #[B]
                    # risk_i = -torch.sum(S_i, dim=1) #[B]
                    # censor = label[:, 9]
                    # survtime = label[:, 11]
                    # if censor.float().mean() != 1:
                    #     cindex_t = CIndex_sksurv(all_risk_scores=risk_t.detach().cpu().numpy().reshape(-1), all_censorships=censor.detach().cpu().numpy().reshape(-1), all_event_times=survtime.detach().cpu().numpy().reshape(-1))
                    #     cindex_i = CIndex_sksurv(all_risk_scores=risk_i.detach().cpu().numpy().reshape(-1), all_censorships=censor.detach().cpu().numpy().reshape(-1), all_event_times=survtime.detach().cpu().numpy().reshape(-1))
                    # else:
                    #     print('\ncensor:', censor)
                    #     print("All samples are censored")
                    #     cindex_t = None
                    #     cindex_i = None
                    score_t = sum([F.softmax(out_t)[i][label[:, 8][i]] for i in range(out_t.size(0))])
                    score_i = sum([F.softmax(out_i)[i][label[:, 8][i]] for i in range(out_i.size(0))])
                elif args.task_type == "grade":
                    score_t = sum([F.softmax(out_t)[i][label[:, 4][i]] for i in range(out_t.size(0))])
                    score_i = sum([F.softmax(out_i)[i][label[:, 4][i]] for i in range(out_i.size(0))])
                elif args.task_type == "subtype":
                    score_t = sum([F.softmax(out_t)[i][label[:, 7][i]] for i in range(out_t.size(0))])
                    score_i = sum([F.softmax(out_i)[i][label[:, 7][i]] for i in range(out_i.size(0))])
                
                
                if args.task_type == 'survival':
                    # if cindex_t is not None and cindex_i is not None:
                    #     ratio_t = cindex_t / cindex_i
                    #     ratio_i = 1 / ratio_t
                    # else:
                    #     ratio_t = None
                    #     ratio_i = None
                    ratio_t = score_t / score_i
                    ratio_i = 1 / ratio_t
                elif args.task_type != 'survival':
                    ratio_t = score_t / score_i
                    ratio_i = 1 / ratio_t
                
                # print('ratio_t:', ratio_t)

                if ratio_t is not None and ratio_i is not None:
                    i_index=0
                    for grad_t, grad_i in zip(model.module.classifier.weight.grad[:, :hs], model.module.classifier.weight.grad[:, hs:]):
                        if grad_t is not None and grad_i is not None:
                            # print('grad_t.shape:', grad_t.shape) #[128]
                            # print('grad_i.shape:', grad_i.shape) #[128]
                            sim = cosine_similarity(grad_t, grad_i)
                            # sim = F.cosine_similarity(grad_t, grad_i)
                            if sim < 0:
                                if ratio_t < 1:
                                    # Calculate the projection of gradient of classifier_tumor onto the direction perpendicular to gradient of classifier
                                    dot_product = torch.dot(grad_t.flatten(), grad_i.flatten())
                                    proj_scale = dot_product / grad_i.norm()**2
                                    proj_component = proj_scale * grad_i
                                    grad_t = grad_t - proj_component
                                    # model.module.classifier.weight.grad[i_index, :hs] = grad_t
                                    perpen = grad_t - proj_component
                                    unit_perpen = perpen / perpen.norm()
                                    grad_t = grad_t.norm() * unit_perpen
                                    model.module.classifier.weight.grad[i_index, :hs] = grad_t
                                elif ratio_i < 1:
                                    # Calculate the projection of gradient of classifier_tumor onto the direction perpendicular to gradient of classifier
                                    dot_product = torch.dot(grad_i.flatten(), grad_t.flatten())
                                    proj_scale = dot_product / grad_t.norm()**2
                                    proj_component = proj_scale * grad_t
                                    grad_i = grad_i - proj_component
                                    # model.module.classifier.weight.grad[i_index, hs:] = grad_i
                                    perpen = grad_i - proj_component
                                    unit_perpen = perpen / perpen.norm()
                                    grad_i = grad_i.norm() * unit_perpen   
                                    model.module.classifier.weight.grad[i_index, hs:] = grad_i
                        i_index = i_index+1
                        
            # Update parameters based on projected gradients
            optimizer.step()

            # if dist.is_available() and dist.is_initialized():
            #     loss = loss.data.clone()
            #     dist.all_reduce(loss.div_(dist.get_world_size()))

            # 2. Synchronize gradients across all processes 同步梯度
            if dist.is_available() and dist.is_initialized():
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad.data,op=dist.ReduceOp.SUM) #Sum gradients
                        p.grad.data /= dist.get_world_size()
            
            cur_iters += 1
            if args.rank == 0: 
                if cur_iters % 10 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    # evaluate on test and val set
                    if args.task_type == "survival":
                        test_cindex = epochVal_survival(model, test_loader, args)
                        if logger is not None and args.return_vgrid:
                            logger.log({'training': {'total loss': loss.item(),
                                                    'batch_sim_loss': batch_sim_loss.item()}})
                            logger.log({'test': {'cindex': test_cindex}})
                        elif logger is not None:
                            logger.log({'training': {'total loss': loss.item()}})
                            logger.log({'test': {'cindex': test_cindex}})
                    else:
                        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochVal(model, test_loader, args)
                        if logger is not None and args.return_vgrid:
                            logger.log({'training': {'total loss': loss.item(),
                                                     'task loss3': loss3.item(),
                                                     'batch_sim_loss': batch_sim_loss.item()}})
                            logger.log({'test': {'Accuracy': test_acc,
                                                    'F1 score': test_f1,
                                                    'AUC': test_auc,
                                                    'Balanced Accuracy': test_bac,
                                                    'Sensitivity': test_sens,
                                                    'Specificity': test_spec,
                                                    'Precision': test_prec}})
                        elif logger is not None:
                            logger.log({'training': {'total loss': loss.item(),
                                                     'task loss3': loss3.item()}})
                            logger.log({'test': {'Accuracy': test_acc,
                                                    'F1 score': test_f1,
                                                    'AUC': test_auc,
                                                    'Balanced Accuracy': test_bac,
                                                    'Sensitivity': test_sens,
                                                    'Specificity': test_spec,
                                                    'Precision': test_prec}})
                    
                    if not args.return_vgrid:
                        print('\rFold: [%2d/%2d] Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                            args.cur_fold, args.kfold, epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item()), end='', flush=True)
                    elif args.return_vgrid:
                        print('\rFold: [%2d/%2d] Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f || Loss3M: %.4f || Lossbb: %.4f' % (
                            args.cur_fold, args.kfold, epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item(), loss3.item(), batch_sim_loss.item()), end='', flush=True)  
        scheduler.step()
        
        # method2: save best model
    if args.rank == 0:
        if args.task_type == "survival":
            test_cindex = epochVal_survival(model, test_loader, args)
            saveModelPath = os.path.join(args.checkpoints, 'fold_{:d}_epoch_{:d}_cindex_{:f}_.pth'.format(
                args.cur_fold, epoch + 1, test_cindex)) 
            if dist.is_available() and dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()        
            torch.save(state_dict, saveModelPath)    
        else: 
            test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochVal(model, test_loader, args)
            saveModelPath = os.path.join(args.checkpoints, 'fold_{:d}_epoch_{:d}_AUC_{:f}_ACC_{:f}_Sens_{:f}_Spec_{:f}_F1_{:f}_.pth'.format(
                args.cur_fold, epoch + 1, test_auc, test_acc, test_sens, test_spec, test_f1))
            if dist.is_available() and dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()        
            torch.save(state_dict, saveModelPath)


# for inference

def testTeachersModel(model, dataloader, logger, args):
    # diag2021_loss_func = nn.CrossEntropyLoss() #
    # 636/1828: (1.0, 4.56, 3.21, 2.65) (636, 288, 409, 495)->( 636+676= 1312, 288, 409, 495)
    # diag2021: (1.0, 4.15, 2.93, 2.43)
    # grad: (1.47, 1.51, 1.0)
    # subtype: (1.0, 1.72, 2.43)

    if args.external_eval:
        diag2021_loss_func = nn.CrossEntropyLoss().cuda()
    else:
        diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    survival_loss_func = NLLSurvLoss(alpha=0.15)
    # distill_loss_func = DistillationLoss(temperature=args.temperature)
    batch_sim_loss_func = PathBatchLoss(args.batch_size, args.world_size) 
    omic_domain_loss_func = OmicDomainScaleLoss(args.batch_size, args.world_size) 
            
    start = time.time()
    _, test_loader = dataloader

    model.eval()  
    with torch.no_grad(): 
        total_loss = 0.0
        if args.save4visualization:
            for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label, wsiID) in enumerate(test_loader):
                x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
                # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time])
                
                feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)

                # classification loss
                #label[:, 5] is for diagnosis
                    
                if args.task_type == "diag2021":
                    taskloss_tea10 = diag2021_loss_func(logits_dict['logits_tea10'].float(), label[:, 5])
                    taskloss_tea20 = diag2021_loss_func(logits_dict['logits_tea20'].float(), label[:, 5])
                    taskloss = taskloss_tea10 + taskloss_tea20
                    # print('testing.........')
                    # taskloss = taskloss_stu10
                    if args.save4roc:
                        # ret.update(objectives.compute_cls(self, batch, test=test))

                        # decoder logits; data labels;
                        prediction_scores_tensors = torch.tensor([]).to(args.device)
                        diag_labels_tensors = torch.tensor([]).to(args.device)
                        # diag_labels_tensors = []

                        prediction_scores_tensors = torch.cat((prediction_scores_tensors, logits_dict['logits_teas']), dim=0)
                        diag_labels_tensors = torch.cat((diag_labels_tensors, label[:, 5]), dim=0)
                        # diag_labels_tensors.append(label[:, 5])

                        prediction_scores_array = prediction_scores_tensors.detach().cpu().numpy()
                        diag_labels_array = diag_labels_tensors.detach().cpu().numpy()
                        # diag_labels_array = np.squeeze(np.array(diag_labels_tensors), axis=0)

                        # print("prediction_scores_array", prediction_scores_array.shape) (16, 1536)
                        # print("diag_labels_array", diag_labels_array.shape) (16)

                        prediction_scores_df = pd.DataFrame(prediction_scores_array)
                        os.makedirs('./save4roc', exist_ok=True)
                        prediction_scores_df.to_csv('./save4roc/teachers_roc_prediction_scores.csv', mode='a', header = False, index = False)

                        diag_labels_df = pd.DataFrame(diag_labels_array)
                        diag_labels_df.to_csv('./save4roc/teachers_roc_diag_labels.csv', mode='a', header = False, index = False)

                    if args.save4visualization:
                        # decoder logits; data labels;
                        wsiIDs = []
                        att1_tea10_tensors = torch.tensor([]).to(args.device)
                        att2_tea10_tensors = torch.tensor([]).to(args.device)
                        prediction_scores_tensors = torch.tensor([]).to(args.device)
                        diag_labels_tensors = torch.tensor([]).to(args.device)
                        # diag_labels_tensors = []

                        # 'att1_tea10'
                        # 'att1_tea20'
                        # 'att2_tea10'
                        # 'att2_tea20'
                        # att1_tea10_scores = att_dict['att1_tea10'].mean(dim=(1, 3))  # From [B,8,2500,144] → [B,2500]
                        # att2_tea10_scores = att_dict['att2_tea10'].mean(dim=(1, 3))  # From [B,8,2500,144] → [B,2500]

                        ## method1
                        att1_tea10_scores, _ = att_dict['att1_tea10'].max(dim=1)  # From [B,8,2500,144] → [B,2500,144] → [B,2500]
                        att2_tea10_scores, _ = att_dict['att2_tea10'].max(dim=1)  # From [B,8,2500,144] → [B,2500,144] → [B,2500]
                        att1_tea10_scores = att1_tea10_scores.mean(dim=2)
                        att2_tea10_scores = att2_tea10_scores.mean(dim=2)
                        ## method2
                        # att1_tea10_scores = att_dict['att1_tea10'][:,1,:,:].mean(dim=2)  # From [B,8,2500,144] → [B,2500,144] → [B,2500]
                        # att2_tea10_scores = att_dict['att2_tea10'][:,1,:,:].mean(dim=2)  # From [B,8,2500,144] → [B,2500,144] → [B,2500]
                        ##
                        # print('att1_tea10_scores.shape:', att1_tea10_scores.shape)

                        wsiIDs.append(wsiID)
                        att1_tea10_tensors = torch.cat((att1_tea10_tensors, att1_tea10_scores), dim=0)
                        att2_tea10_tensors = torch.cat((att2_tea10_tensors, att2_tea10_scores), dim=0)
                        prediction_scores_tensors = torch.cat((prediction_scores_tensors, logits_dict['logits_teas']), dim=0)
                        diag_labels_tensors = torch.cat((diag_labels_tensors, label[:, 5]), dim=0)
                        # diag_labels_tensors.append(label[:, 5])

                        att1_tea10_array = att1_tea10_tensors.detach().cpu().numpy()
                        att2_tea10_array = att2_tea10_tensors.detach().cpu().numpy()
                        prediction_scores_array = prediction_scores_tensors.detach().cpu().numpy()
                        diag_labels_array = diag_labels_tensors.detach().cpu().numpy()
                        # diag_labels_array = np.squeeze(np.array(diag_labels_tensors), axis=0)

                        # print("prediction_scores_array", prediction_scores_array.shape) (16, 1536)
                        # print("diag_labels_array", diag_labels_array.shape) (16)
                        os.makedirs('./outputs', exist_ok=True)

                        att1_tea10_df = pd.DataFrame(att1_tea10_array)
                        # att1_tea10_df.to_csv('./outputs/teachers_att1_tea10.csv', mode='a', header = False, index = False) 

                        att2_tea10_df = pd.DataFrame(att2_tea10_array)
                        # att2_tea10_df.to_csv('./outputs/teachers_att2_tea10.csv', mode='a', header = False, index = False) 

                        prediction_scores_df = pd.DataFrame(prediction_scores_array)
                        # prediction_scores_df.to_csv('./outputs/teachers_prediction_scores.csv', mode='a', header = False, index = False)

                        diag_labels_df = pd.DataFrame(diag_labels_array)
                        # diag_labels_df.to_csv('./outputs/teachers_diag_labels.csv', mode='a', header = False, index = False) 

                        # pd.DataFrame(wsiIDs).to_csv('./outputs/teachers_wsiID.csv', mode='a', header=False, index=False)        

                elif args.task_type == "survival":
                    taskloss_tea10 = survival_loss_func(hazards=hazards_dict['hazards_tea10'], S=S_dict['S_tea10'], Y=label[:,8], c=label[:,9], alpha=0)
                    taskloss_tea20 = survival_loss_func(hazards=hazards_dict['hazards_tea20'], S=S_dict['S_tea20'], Y=label[:,8], c=label[:,9], alpha=0)
                    taskloss = taskloss_tea10 + taskloss_tea20
                elif args.task_type == "grade":
                    taskloss_tea10 = grade_loss_func(logits_dict['logits_tea10'].float(), label[:, 4])
                    taskloss_tea20 = grade_loss_func(logits_dict['logits_tea20'].float(), label[:, 4])
                    taskloss = taskloss_tea10 + taskloss_tea20
                elif args.task_type == "subtype":
                    taskloss_tea10 = subtype_loss_func(logits_dict['logits_tea10'].float(), label[:, 7])
                    taskloss_tea20 = subtype_loss_func(logits_dict['logits_tea20'].float(), label[:, 7])
                    taskloss = taskloss_tea10 + taskloss_tea20
                
                # for distillation loss
                # distillloss = distill_loss_func(logits_dict['logits_tea20'], logits_dict['logits_tea10'])

                # if args.multiscale_attention and epoch>1:
                # if args.multiscale_attention:
                #     batchloss_mole = torch.sum(omic_domain_loss_func(att_dict['att1_tea10'], att_dict['att1_tea20'],
                #                                                         att_dict['att2_tea10'], att_dict['att2_tea20']))
                #     batchloss = batchloss_mole                
                # # loss1 = diag2021_loss_func(logits[0], label[:, 5])
                # # loss2 = diag2021_loss_func(logits[1], label[:, 5])
                # if args.multiscale_attention:
                #     loss = taskloss + batchloss
                #     # loss = batchloss #for batchloss debug!!!
                # else:
                #     loss = taskloss  

                loss = taskloss

                total_loss += loss.item() 

                print('\rTest Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' % (
                    i + 1, len(test_loader), time.time() - start, loss.item()), end='', flush=True)
        else:
            for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(test_loader):
                x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
                # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time])
                
                feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)

                # classification loss
                #label[:, 5] is for diagnosis
                    
                if args.task_type == "diag2021":
                    taskloss_tea10 = diag2021_loss_func(logits_dict['logits_tea10'].float(), label[:, 5])
                    taskloss_tea20 = diag2021_loss_func(logits_dict['logits_tea20'].float(), label[:, 5])
                    taskloss = taskloss_tea10 + taskloss_tea20
                    # print('testing.........')
                    # taskloss = taskloss_stu10
                    if args.save4roc:
                        # ret.update(objectives.compute_cls(self, batch, test=test))

                        # decoder logits; data labels;
                        prediction_scores_tensors = torch.tensor([]).to(args.device)
                        diag_labels_tensors = torch.tensor([]).to(args.device)
                        # diag_labels_tensors = []

                        prediction_scores_tensors = torch.cat((prediction_scores_tensors, logits_dict['logits_teas']), dim=0)
                        diag_labels_tensors = torch.cat((diag_labels_tensors, label[:, 5]), dim=0)
                        # diag_labels_tensors.append(label[:, 5])

                        prediction_scores_array = prediction_scores_tensors.detach().cpu().numpy()
                        diag_labels_array = diag_labels_tensors.detach().cpu().numpy()
                        # diag_labels_array = np.squeeze(np.array(diag_labels_tensors), axis=0)

                        # print("prediction_scores_array", prediction_scores_array.shape) (16, 1536)
                        # print("diag_labels_array", diag_labels_array.shape) (16)

                        prediction_scores_df = pd.DataFrame(prediction_scores_array)
                        os.makedirs('./save4roc', exist_ok=True)
                        prediction_scores_df.to_csv('./save4roc/teachers_roc_prediction_scores.csv', mode='a', header = False, index = False)

                        diag_labels_df = pd.DataFrame(diag_labels_array)
                        diag_labels_df.to_csv('./save4roc/teachers_roc_diag_labels.csv', mode='a', header = False, index = False)

                elif args.task_type == "survival":
                    taskloss_tea10 = survival_loss_func(hazards=hazards_dict['hazards_tea10'], S=S_dict['S_tea10'], Y=label[:,8], c=label[:,9], alpha=0)
                    taskloss_tea20 = survival_loss_func(hazards=hazards_dict['hazards_tea20'], S=S_dict['S_tea20'], Y=label[:,8], c=label[:,9], alpha=0)
                    taskloss = taskloss_tea10 + taskloss_tea20
                elif args.task_type == "grade":
                    taskloss_tea10 = grade_loss_func(logits_dict['logits_tea10'].float(), label[:, 4])
                    taskloss_tea20 = grade_loss_func(logits_dict['logits_tea20'].float(), label[:, 4])
                    taskloss = taskloss_tea10 + taskloss_tea20
                elif args.task_type == "subtype":
                    taskloss_tea10 = subtype_loss_func(logits_dict['logits_tea10'].float(), label[:, 7])
                    taskloss_tea20 = subtype_loss_func(logits_dict['logits_tea20'].float(), label[:, 7])
                    taskloss = taskloss_tea10 + taskloss_tea20
                
                # for distillation loss
                # distillloss = distill_loss_func(logits_dict['logits_tea20'], logits_dict['logits_tea10'])

                # if args.multiscale_attention and epoch>1:
                # if args.multiscale_attention:
                #     batchloss_mole = torch.sum(omic_domain_loss_func(att_dict['att1_tea10'], att_dict['att1_tea20'],
                #                                                         att_dict['att2_tea10'], att_dict['att2_tea20']))
                #     batchloss = batchloss_mole                
                # # loss1 = diag2021_loss_func(logits[0], label[:, 5])
                # # loss2 = diag2021_loss_func(logits[1], label[:, 5])
                # if args.multiscale_attention:
                #     loss = taskloss + batchloss
                #     # loss = batchloss #for batchloss debug!!!
                # else:
                #     loss = taskloss  

                loss = taskloss

                total_loss += loss.item() 

                print('\rTest Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' % (
                    i + 1, len(test_loader), time.time() - start, loss.item()), end='', flush=True)

        #next section
        avg_loss = total_loss / len(test_loader)
        if logger is not None:
            logger.log({'test': {'Average Loss': avg_loss}})

        # evaluate on test and val set
        if args.task_type == "survival":
            test_cindex_dict = epochDistillVal_survival(model, test_loader, args, model_type='teacher')
            if logger is not None and args.multiscale_attention:
                logger.log({'test': {'cindex_teas': test_cindex_dict['cindex_teas'],
                                        'cindex_tea10': test_cindex_dict['cindex_tea10'],
                                        'cindex_tea20': test_cindex_dict['cindex_tea20'],}})
            elif logger is not None:
                logger.log({'test': {'cindex_teas': test_cindex_dict['cindex_teas'],
                                        'cindex_tea10': test_cindex_dict['cindex_tea10'],
                                        'cindex_tea20': test_cindex_dict['cindex_tea20'],}})
        else:
            test_acc_dict, test_f1_dict, test_auc_dict, test_bac_dict, test_sens_dict, test_spec_dict, test_prec_dict = epochDistillVal(model, test_loader, args, model_type='teacher')
            if logger is not None and args.multiscale_attention:
                logger.log({'test_teas': {'Accuracy': test_acc_dict['acc_teas'],
                                        'F1 score': test_f1_dict['f1_teas'],
                                        'AUC': test_auc_dict['auc_teas'],
                                        'Balanced Accuracy': test_bac_dict['bac_teas'],
                                        'Sensitivity': test_sens_dict['sens_teas'],
                                        'Specificity': test_spec_dict['spec_teas'],
                                        'Precision': test_prec_dict['prec_teas']},
                            'test_tea10': {'Accuracy': test_acc_dict['acc_tea10'],
                                        'F1 score': test_f1_dict['f1_tea10'],
                                        'AUC': test_auc_dict['auc_tea10'],
                                        'Balanced Accuracy': test_bac_dict['bac_tea10'],
                                        'Sensitivity': test_sens_dict['sens_tea10'],
                                        'Specificity': test_spec_dict['spec_tea10'],
                                        'Precision': test_prec_dict['prec_tea10']},
                            'test_tea20': {'Accuracy': test_acc_dict['acc_tea20'],
                                        'F1 score': test_f1_dict['f1_tea20'],
                                        'AUC': test_auc_dict['auc_tea20'],
                                        'Balanced Accuracy': test_bac_dict['bac_tea20'],
                                        'Sensitivity': test_sens_dict['sens_tea20'],
                                        'Specificity': test_spec_dict['spec_tea20'],
                                        'Precision': test_prec_dict['prec_tea20']},
                                        })
            elif logger is not None:
                logger.log({'test_teas': {'Accuracy': test_acc_dict['acc_teas'],
                                        'F1 score': test_f1_dict['f1_teas'],
                                        'AUC': test_auc_dict['auc_teas'],
                                        'Balanced Accuracy': test_bac_dict['bac_teas'],
                                        'Sensitivity': test_sens_dict['sens_teas'],
                                        'Specificity': test_spec_dict['spec_teas'],
                                        'Precision': test_prec_dict['prec_teas']},
                            'test_tea10': {'Accuracy': test_acc_dict['acc_tea10'],
                                        'F1 score': test_f1_dict['f1_tea10'],
                                        'AUC': test_auc_dict['auc_tea10'],
                                        'Balanced Accuracy': test_bac_dict['bac_tea10'],
                                        'Sensitivity': test_sens_dict['sens_tea10'],
                                        'Specificity': test_spec_dict['spec_tea10'],
                                        'Precision': test_prec_dict['prec_tea10']},
                            'test_tea20': {'Accuracy': test_acc_dict['acc_tea20'],
                                        'F1 score': test_f1_dict['f1_tea20'],
                                        'AUC': test_auc_dict['auc_tea20'],
                                        'Balanced Accuracy': test_bac_dict['bac_tea20'],
                                        'Sensitivity': test_sens_dict['sens_tea20'],
                                        'Specificity': test_spec_dict['spec_tea20'],
                                        'Precision': test_prec_dict['prec_tea20']},
                                        })
    print("\nTesting completed. Average Loss: {:.4f} || cindex_teas: {:.4f} \n".format(avg_loss, test_cindex_dict['cindex_teas']))

def testStudentsModel(model, dataloader, logger, args):
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    survival_loss_func = NLLSurvLoss(alpha=0.15)
    # distill_loss_func = DistillationLoss(temperature=args.temperature)
    batch_sim_loss_func = PathBatchLoss(args.batch_size, args.world_size) 
    omic_domain_loss_func = OmicDomainScaleLoss(args.batch_size, args.world_size) 
            
    start = time.time()
    model.eval()
    _, test_loader = dataloader

    model.eval()  
    with torch.no_grad(): 
        total_loss = 0.0
        for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(test_loader):
            x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time])
            
            feature_dict, att_dict, logits_dict, hazards_dict, S_dict, risk_dict = model(x_path10=x_path10, x_path20=x_path20)

            # classification loss
            #label[:, 5] is for diagnosis
                
            if args.task_type == "diag2021":
                taskloss_stu10 = diag2021_loss_func(logits_dict['logits_stu10'].float(), label[:, 5])
                taskloss_stu20 = diag2021_loss_func(logits_dict['logits_stu20'].float(), label[:, 5])
                taskloss = taskloss_stu10 + taskloss_stu20
                # print('testing.........')
                # taskloss = taskloss_stu10

            elif args.task_type == "survival":
                taskloss_stu10 = survival_loss_func(hazards=hazards_dict['hazards_stu10'], S=S_dict['S_stu10'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss_stu20 = survival_loss_func(hazards=hazards_dict['hazards_stu20'], S=S_dict['S_stu20'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss = taskloss_stu10 + taskloss_stu20
            elif args.task_type == "grade":
                taskloss_stu10 = grade_loss_func(logits_dict['logits_stu10'].float(), label[:, 4])
                taskloss_stu20 = grade_loss_func(logits_dict['logits_stu20'].float(), label[:, 4])
                taskloss = taskloss_stu10 + taskloss_stu20
            elif args.task_type == "subtype":
                taskloss_stu10 = subtype_loss_func(logits_dict['logits_stu10'].float(), label[:, 7])
                taskloss_stu20 = subtype_loss_func(logits_dict['logits_stu20'].float(), label[:, 7])
                taskloss = taskloss_stu10 + taskloss_stu20

            # # if args.multiscale_attention and epoch>1:
            # if args.multiscale_attention:
            #     batchloss_histo = torch.sum(batch_sim_loss_func(att_dict['att_stu10'], att_dict['att_stu20']))
            #     # batch_sim_loss = 0.5*batch_sim_loss_path + 0.5*batch_sim_loss_omic  
            #     batchloss = 1000*batchloss_histo             
            # if args.multiscale_attention:
            #     loss = taskloss + batchloss
            # else:
            #     loss = taskloss  

            loss = taskloss

            total_loss += loss.item() 

            print('\rTest Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' % (
                i + 1, len(test_loader), time.time() - start, loss.item()), end='', flush=True)
        
        avg_loss = total_loss / len(test_loader)
        if logger is not None:
            logger.log({'test': {'Average Loss': avg_loss}})

        if args.task_type == "survival":
            test_cindex_dict = epochDistillVal_survival(model, test_loader, args, model_type='student')
            if logger is not None and args.multiscale_attention:
                logger.log({'test': {'cindex_stu10': test_cindex_dict['cindex_stu10'],
                                        'cindex_stu20': test_cindex_dict['cindex_stu20'],
                                        'cindex_stus': test_cindex_dict['cindex_stus'],}})
            elif logger is not None:
                logger.log({'test': {'cindex_stu10': test_cindex_dict['cindex_stu10'],
                                        'cindex_stu20': test_cindex_dict['cindex_stu20'],
                                        'cindex_stus': test_cindex_dict['cindex_stus'],}})
        else:
            test_acc_dict, test_f1_dict, test_auc_dict, test_bac_dict, test_sens_dict, test_spec_dict, test_prec_dict = epochDistillVal(model, test_loader, args, model_type='student')
            if logger is not None and args.multiscale_attention:
                logger.log({'test_stu10': {'Accuracy': test_acc_dict['acc_stu10'],
                                        'F1 score': test_f1_dict['f1_stu10'],
                                        'AUC': test_auc_dict['auc_stu10'],
                                        'Balanced Accuracy': test_bac_dict['bac_stu10'],
                                        'Sensitivity': test_sens_dict['sens_stu10'],
                                        'Specificity': test_spec_dict['spec_stu10'],
                                        'Precision': test_prec_dict['prec_stu10']},
                            'test_stu20': {'Accuracy': test_acc_dict['acc_stu20'],
                                        'F1 score': test_f1_dict['f1_stu20'],
                                        'AUC': test_auc_dict['auc_stu20'],
                                        'Balanced Accuracy': test_bac_dict['bac_stu20'],
                                        'Sensitivity': test_sens_dict['sens_stu20'],
                                        'Specificity': test_spec_dict['spec_stu20'],
                                        'Precision': test_prec_dict['prec_stu20']},
                            'test_stus': {'Accuracy': test_acc_dict['acc_stus'],
                                        'F1 score': test_f1_dict['f1_stus'],
                                        'AUC': test_auc_dict['auc_stus'],
                                        'Balanced Accuracy': test_bac_dict['bac_stus'],
                                        'Sensitivity': test_sens_dict['sens_stus'],
                                        'Specificity': test_spec_dict['spec_stus'],
                                        'Precision': test_prec_dict['prec_stus']},
                                        })
            elif logger is not None:
                logger.log({'test_stu10': {'Accuracy': test_acc_dict['acc_stu10'],
                                        'F1 score': test_f1_dict['f1_stu10'],
                                        'AUC': test_auc_dict['auc_stu10'],
                                        'Balanced Accuracy': test_bac_dict['bac_stu10'],
                                        'Sensitivity': test_sens_dict['sens_stu10'],
                                        'Specificity': test_spec_dict['spec_stu10'],
                                        'Precision': test_prec_dict['prec_stu10']},
                            'test_stu20': {'Accuracy': test_acc_dict['acc_stu20'],
                                        'F1 score': test_f1_dict['f1_stu20'],
                                        'AUC': test_auc_dict['auc_stu20'],
                                        'Balanced Accuracy': test_bac_dict['bac_stu20'],
                                        'Sensitivity': test_sens_dict['sens_stu20'],
                                        'Specificity': test_spec_dict['spec_stu20'],
                                        'Precision': test_prec_dict['prec_stu20']},
                            'test_stus': {'Accuracy': test_acc_dict['acc_stus'],
                                        'F1 score': test_f1_dict['f1_stus'],
                                        'AUC': test_auc_dict['auc_stus'],
                                        'Balanced Accuracy': test_bac_dict['bac_stus'],
                                        'Sensitivity': test_sens_dict['sens_stus'],
                                        'Specificity': test_spec_dict['spec_stus'],
                                        'Precision': test_prec_dict['prec_stus']},
                                        })
                    
    print("\nTesting completed. Average Loss: {:.4f} || cindex_stus: {:.4f} \n".format(avg_loss, test_cindex_dict['cindex_stus']))


def testDistillation(student_model, teacher_model, dataloader, logger, args):
    student_model.eval()
    teacher_model.eval()
    # diag2021_loss_func = nn.CrossEntropyLoss() #
    # 636/1828: (1.0, 4.56, 3.21, 2.65) (636, 288, 409, 495)->( 636+676= 1312, 288, 409, 495)
    # diag2021: (1.0, 4.15, 2.93, 2.43)
    # grad: (1.47, 1.51, 1.0)
    # subtype: (1.0, 1.72, 2.43)
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    survival_loss_func = NLLSurvLoss(alpha=0.15)
    distill_loss_func = DistillationLoss(temperature=args.temperature)
    batch_sim_loss_func = PathBatchLoss(args.batch_size, args.world_size) 
            
    start = time.time()

    _, test_loader = dataloader

    with torch.no_grad(): 
        total_loss = 0.0
        for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(test_loader):
            x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label = x_path10.cuda(), x_path20.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor, 10:label_event, 11:survival_time])
            
            feature_dict_tea, att_dict_tea, logits_dict_tea, hazards_dict_tea, S_dict_tea, risk_dict_tea = teacher_model(x_path10=x_path10, x_path20=x_path20, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            feature_tea10 = torch.cat((feature_dict_tea['feature1_tea10'], feature_dict_tea['feature2_tea10']), dim=-1)
            feature_tea20 = torch.cat((feature_dict_tea['feature1_tea20'], feature_dict_tea['feature2_tea20']), dim=-1)

            # Get student outputs
            feature_dict_stu, att_dict_stu, logits_dict_stu, hazards_dict_stu, S_dict_stu, risk_dict_stu = student_model(x_path10=x_path10, x_path20=x_path20)
            feature_stu10 = feature_dict_stu['feature_stu10']
            feature_stu20 = feature_dict_stu['feature_stu20']
                
            if args.task_type == "diag2021":
                taskloss_stu10 = diag2021_loss_func(logits_dict_stu['logits_stu10'].float(), label[:, 5])
                taskloss_stu20 = diag2021_loss_func(logits_dict_stu['logits_stu20'].float(), label[:, 5])
                taskloss = taskloss_stu10 + taskloss_stu20
                # print('testing.........')
                # taskloss = taskloss_stu10

            elif args.task_type == "survival":
                taskloss_stu10 = survival_loss_func(hazards=hazards_dict_stu['hazards_stu10'], S=S_dict_stu['S_stu10'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss_stu20 = survival_loss_func(hazards=hazards_dict_stu['hazards_stu20'], S=S_dict_stu['S_stu20'], Y=label[:,8], c=label[:,9], alpha=0)
                taskloss = taskloss_stu10 + taskloss_stu20
            elif args.task_type == "grade":
                taskloss_stu10 = grade_loss_func(logits_dict_stu['logits_stu10'].float(), label[:, 4])
                taskloss_stu20 = grade_loss_func(logits_dict_stu['logits_stu20'].float(), label[:, 4])
                taskloss = taskloss_stu10 + taskloss_stu20
            elif args.task_type == "subtype":
                taskloss_stu10 = subtype_loss_func(logits_dict_stu['logits_stu10'].float(), label[:, 7])
                taskloss_stu20 = subtype_loss_func(logits_dict_stu['logits_stu20'].float(), label[:, 7])
                taskloss = taskloss_stu10 + taskloss_stu20
            
            # for distillation loss
            if args.distill_logits:
                distillloss_logits_10 = distill_loss_func(logits_dict_stu['logits_stu10'], logits_dict_tea['logits_tea10'])
                distillloss_logits_20 = distill_loss_func(logits_dict_stu['logits_stu20'], logits_dict_tea['logits_tea20'])
                distillloss_logits = distillloss_logits_10 + distillloss_logits_20
            if args.distill_feature:
                distillloss_feature_10 = F.mse_loss(feature_stu10, feature_tea10)
                distillloss_feature_20 = F.mse_loss(feature_stu20, feature_tea20)
                distillloss_feature = distillloss_feature_10 + distillloss_feature_20

            # if args.multiscale_attention:
            #     batchloss_histo = torch.sum(batch_sim_loss_func(att_dict_stu['att_stu10'], att_dict_stu['att_stu20']))
            #     # batch_sim_loss = 0.5*batch_sim_loss_path + 0.5*batch_sim_loss_omic  
            #     batchloss = batchloss_histo           
            # if args.distill_logits and args.distill_feature:
            #     loss = taskloss + distillloss_logits + distillloss_feature
            # elif args.distill_logits and not args.distill_feature:
            #     loss = taskloss + distillloss_logits
            # elif not args.distill_logits and args.distill_feature:
            #     loss = taskloss + distillloss_feature
            # else:
            #     loss = taskloss  

            loss = taskloss
            
            total_loss += loss.item() 

            print('\rTest Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' % (
                i + 1, len(test_loader), time.time() - start, loss.item()), end='', flush=True)
        
        avg_loss = total_loss / len(test_loader)
        if logger is not None:
            logger.log({'test': {'Average Loss': avg_loss}})

        if args.task_type == "survival":
            test_cindex_dict_stu = epochDistillVal_survival(student_model, test_loader, args, model_type='student')
            if logger is not None and args.multiscale_attention:
                logger.log({'test': {'cindex_stu10': test_cindex_dict_stu['cindex_stu10'],
                                        'cindex_stu20': test_cindex_dict_stu['cindex_stu20'],
                                        'cindex_stus': test_cindex_dict_stu['cindex_stus'],}})
            elif logger is not None:
                logger.log({'test': {'cindex_stu10': test_cindex_dict_stu['cindex_stu10'],
                                        'cindex_stu20': test_cindex_dict_stu['cindex_stu20'],
                                        'cindex_stus': test_cindex_dict_stu['cindex_stus'],}})
        else:
            test_acc_dict_stu, test_f1_dict_stu, test_auc_dict_stu, test_bac_dict_stu, test_sens_dict_stu, test_spec_dict_stu, test_prec_dict_stu = epochDistillVal(student_model, test_loader, args, model_type='student')
            if logger is not None and args.multiscale_attention:
                logger.log({'test_stu10': {'Accuracy': test_acc_dict_stu['acc_stu10'],
                                        'F1 score': test_f1_dict_stu['f1_stu10'],
                                        'AUC': test_auc_dict_stu['auc_stu10'],
                                        'Balanced Accuracy': test_bac_dict_stu['bac_stu10'],
                                        'Sensitivity': test_sens_dict_stu['sens_stu10'],
                                        'Specificity': test_spec_dict_stu['spec_stu10'],
                                        'Precision': test_prec_dict_stu['prec_stu10']},
                            'test_stu20': {'Accuracy': test_acc_dict_stu['acc_stu20'],
                                        'F1 score': test_f1_dict_stu['f1_stu20'],
                                        'AUC': test_auc_dict_stu['auc_stu20'],
                                        'Balanced Accuracy': test_bac_dict_stu['bac_stu20'],
                                        'Sensitivity': test_sens_dict_stu['sens_stu20'],
                                        'Specificity': test_spec_dict_stu['spec_stu20'],
                                        'Precision': test_prec_dict_stu['prec_stu20']},
                            'test_stus': {'Accuracy': test_acc_dict_stu['acc_stus'],
                                        'F1 score': test_f1_dict_stu['f1_stus'],
                                        'AUC': test_auc_dict_stu['auc_stus'],
                                        'Balanced Accuracy': test_bac_dict_stu['bac_stus'],
                                        'Sensitivity': test_sens_dict_stu['sens_stus'],
                                        'Specificity': test_spec_dict_stu['spec_stus'],
                                        'Precision': test_prec_dict_stu['prec_stus']},
                                        })
            elif logger is not None:
                logger.log({'test_stu10': {'Accuracy': test_acc_dict_stu['acc_stu10'],
                                        'F1 score': test_f1_dict_stu['f1_stu10'],
                                        'AUC': test_auc_dict_stu['auc_stu10'],
                                        'Balanced Accuracy': test_bac_dict_stu['bac_stu10'],
                                        'Sensitivity': test_sens_dict_stu['sens_stu10'],
                                        'Specificity': test_spec_dict_stu['spec_stu10'],
                                        'Precision': test_prec_dict_stu['prec_stu10']},
                            'test_stu20': {'Accuracy': test_acc_dict_stu['acc_stu20'],
                                        'F1 score': test_f1_dict_stu['f1_stu20'],
                                        'AUC': test_auc_dict_stu['auc_stu20'],
                                        'Balanced Accuracy': test_bac_dict_stu['bac_stu20'],
                                        'Sensitivity': test_sens_dict_stu['sens_stu20'],
                                        'Specificity': test_spec_dict_stu['spec_stu20'],
                                        'Precision': test_prec_dict_stu['prec_stu20']},
                            'test_stus': {'Accuracy': test_acc_dict_stu['acc_stus'],
                                        'F1 score': test_f1_dict_stu['f1_stus'],
                                        'AUC': test_auc_dict_stu['auc_stus'],
                                        'Balanced Accuracy': test_bac_dict_stu['bac_stus'],
                                        'Sensitivity': test_sens_dict_stu['sens_stus'],
                                        'Specificity': test_spec_dict_stu['spec_stus'],
                                        'Precision': test_prec_dict_stu['prec_stus']},
                                        })
                    
    print("\nTesting completed. Average Loss: {:.4f} || cindex_stus: {:.4f} \n".format(avg_loss, test_cindex_dict_stu['cindex_stus']))

def testDeformPathomicModel(model, dataloader, logger, args):
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    # survival_loss_func = NLLSurvLoss(alpha=0.0)
    nll_loss_func = NLLSurvLoss(alpha=0.15)
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)  
            
    start = time.time()
    model.eval()
    _, test_loader = dataloader 
    with torch.no_grad(): 
        total_loss = 0.0
        for i, (x_path, _, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(test_loader):
        # for i, (x_path10, x_path20, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(train_loader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda().long()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor])
            
            # features, path_vec, omic_vec, logits, pred, pred_path, pred_omic, fuse_grads, path_grads, omic_grads
            # fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            # print('logits[2].shape:', logits[2].shape)#[8,4]
            S = torch.cumprod(1 - logits[2], dim=1)
            # print('S.shape:', S.shape) #[8,4]

            # classification loss
            # loss = diag2021_loss_func(logits[2], label[:, 5])
            # print('label.shape:', label.shape) #torch.Size([2, 7], 
            # logits[2] for tumor+immune diagnosis; label[:, 5] is for diagnosis
                
            if args.task_type == "diag2021":
                loss3 = diag2021_loss_func(logits[2], label[:, 5])
            elif args.task_type == "survival":
                hazard_pred = logits[2]
                loss_nll = nll_loss_func(hazards=hazard_pred, S=S, Y=label[:,8], c=label[:,9], alpha=0)
                loss3 = loss_nll

            # elif args.task_type == "grade":
            #     loss3 = grade_loss_func(logits[2], label[:, 4])
            # elif args.task_type == "subtype":
            #     loss3 = subtype_loss_func(logits[2], label[:, 7])
            # if args.return_vgrid:
            #     batch_sim_loss_tumor = torch.sum(batch_sim_loss_func(logits[3], logits[4]))
            #     batch_sim_loss_immune = torch.sum(batch_sim_loss_func(logits[5], logits[6]))
            #     batch_sim_loss = 0.5*batch_sim_loss_tumor + 0.5*batch_sim_loss_immune
            #     loss = loss3+batch_sim_loss
            # else:
            #     loss = loss3  

            loss = loss3
            total_loss += loss.item() 

            print('\rTest Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' % (
                i + 1, len(test_loader), time.time() - start, loss.item()), end='', flush=True)
        
        avg_loss = total_loss / len(test_loader)
        if logger is not None:
            logger.log({'test': {'Average Loss': avg_loss}})

        if args.task_type == "survival":
            test_cindex = epochVal_survival(model, test_loader, args)
            if logger is not None and args.return_vgrid:
                logger.log({'test': {'cindex': test_cindex}})
            elif logger is not None:
                logger.log({'training': {'total loss': loss.item()}})
                logger.log({'test': {'cindex': test_cindex}})
        else:
            test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochVal(model, test_loader, args)
            if logger is not None and args.return_vgrid:
                logger.log({'test': {'Accuracy': test_acc,
                                        'F1 score': test_f1,
                                        'AUC': test_auc,
                                        'Balanced Accuracy': test_bac,
                                        'Sensitivity': test_sens,
                                        'Specificity': test_spec,
                                        'Precision': test_prec}})
            elif logger is not None:
                logger.log({'test': {'Accuracy': test_acc,
                                        'F1 score': test_f1,
                                        'AUC': test_auc,
                                        'Balanced Accuracy': test_bac,
                                        'Sensitivity': test_sens,
                                        'Specificity': test_spec,
                                        'Precision': test_prec}})

    print("\nTesting completed. Average Loss: {:.4f} || cindex_stus: {:.4f} \n".format(avg_loss, test_cindex))                    

     


            