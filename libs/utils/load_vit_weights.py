psa_n_laelif task == 'vqa_with_test':
        vqa_val_ques=os.path.join(vqa_dir,'val_questions.json')
        vqa_val_ann=os.path.join(vqa_dir,'val_annotations.json')
        vqa_test_ques=os.path.join(vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')
        vqa_test_ann=os.path.join(vqa_dir,'v2_mscoco_val2014_annotations.json')
        vocab_file=os.path.join('conf/vqa_vocab.pkl')
        with open(vocab_file,'rb') as f:
                ans_to_id,id_to_ans=pickle.loads(f.read())
        vqa_val = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)
        vqa_test = VQA(annotation_file=vqa_test_ann, question_file=vqa_test_ques)
        DL,val_dl,test_dl,val_non_ma_dl = dl.vqa_batchgen_with_test(vqa_dir, coco_images, num_workers=args.n_workers, batch_size=batch_size, data_seed=int(args.data_seed+restore))
        if args.optim == 'adam':
            if args.switch_cca_omninet_freq != 0:
                layer_switch(shared_model, layers_cca, freeze=True)
            omni_optimizer = ScheduledOptim(
                Adam(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr,name='optimizer')
        elif args.optim == 'adamw':
            omni_optimizer = ScheduledOptim(
                AdamW(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr,name='optimizer')
        optimizer = omni_optimizer
        if os.path.exists(os.path.join(args.model_save_path,str(restore),'optimizer.pth')):
            optimizer.restore(args.model_save_path, restore)
        if args.switch_cca_omninet_freq != 0:
            layer_switch(shared_model, layers_cca, freeze=False)
            layer_switch(shared_model, layers_omninet, freeze=True)
            cca_optimizer = ScheduledOptim(
                Adam(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay_cca),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr_cca,name='cca_optimizer')
            if os.path.exists(os.path.join(args.model_save_path,str(restore),'cca_optimizer.pth')):
                cca_optimizer.restore(args.model_save_path, restore)
            if os.path.exists(os.path.join(args.model_save_path,str(restore),'omni_on.pkl')):
                with open(os.path.join(args.model_save_path,str(restore),'omni_on.pkl'), 'rb') as f:
                    omni_on = pickle.load(f)
            if not omni_on:
                optimizer = cca_optimizer
    elif task == 'vqa_metric':
        # vqa_train_ques=os.path.join(vqa_dir,'metric_experiment_train_questions.json')
        # vqa_train_ann=os.path.join(vqa_dir,'metric_experiment_train_annotations.json')
        vqa_val_ques=os.path.join(vqa_dir,'metric_experiment_val_questions.json')
        vqa_val_ann=os.path.join(vqa_dir,'metric_experiment_val_annotations.json')
        vqa_test_ques=os.path.join(vqa_dir,'metric_experiment_test_questions.json')
        vqa_test_ann=os.path.join(vqa_dir,'metric_experiment_test_annotations.json')
        vocab_file=os.path.join('conf/vqa_vocab.pkl')
        with open(vocab_file,'rb') as f:
                ans_to_id,id_to_ans=pickle.loads(f.read())
        vqa_val = VQA(annotation_file=vqa_val_ann, question_file=vqa_val_ques)
        vqa_test = VQA(annotation_file=vqa_test_ann, question_file=vqa_test_ques)

        DL,val_dl,test_dl = dl.vqa_metric_exp_batchgen(vqa_dir, coco_images, num_workers=args.n_workers, batch_size=batch_size, data_seed=int(args.data_seed+restore))
        if args.optim == 'adam':
            if args.switch_cca_omninet_freq != 0:
                layer_switch(shared_model, layers_cca, freeze=True)
            omni_optimizer = ScheduledOptim(
                Adam(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr,name='optimizer')
        elif args.optim == 'adamw':
            omni_optimizer = ScheduledOptim(
                AdamW(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr,name='optimizer')
        optimizer = omni_optimizer
        if os.path.exists(os.path.join(args.model_save_path,str(restore),'optimizer.pth')):
            optimizer.restore(args.model_save_path, restore)
        if args.switch_cca_omninet_freq != 0:
            layer_switch(shared_model, layers_cca, freeze=False)
            layer_switch(shared_model, layers_omninet, freeze=True)
            cca_optimizer = ScheduledOptim(
                Adam(
                    filter(lambda x: x.requires_grad, shared_model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay_cca),
                512, 16000,restore,max_lr=0.0001,init_lr=args.init_lr_cca,name='cca_optimizer')
            if os.path.exists(os.path.join(args.model_save_path,str(restore),'cca_optimizer.pth')):
                cca_optimizer.restore(args.model_save_path, restore)
            if os.path.exists(os.path.join(args.model_save_path,str(restore),'omni_on.pkl')):
                with open(os.path.join(args.model_save_path,str(restore),'omni_on.pkl'), 'rb') as f:
                    omni_on = pickle.load(f)
            if not omni_on:
                optimizer = cca_optimizer
               elif task == 'vq elif task == 'vqa' or task == 'vqa_metric' or task == 'vqa_with_test':if args.save_decoder_attn or args.save_cca_attn: #len(vqa_out) > 4:ques_id = b['ques_id']
                        vocab_file=os.path.join('conf/vqa_vocab.pkl')
                        with open(vocab_file,'rb') as f:
                            ans_to_id,id_to_ans=pickle.loads(f.read())
                        ytrue = torch.reshape(answers, [-1]).cpu().numpy()
                        preds = pred.detach().cpu().squeeze().tolist()
                        preds_str = [id_to_ans[pr] for pr in preds]
                        ytrue_str = [id_to_ans[yt] for yt in ytrue]
                        if args.save_decoder_attn: #len(vqa_out) == 6:
                            dec_attns = vqa_out[5]
                            write_attn(args.model_save_path+'_dec_attns_val_ma', {'dec_spat_attn': dec_attns[1].detach().cpu().numpy(), 'dec_tmp_attn': dec_attns[2].detach().cpu().numpy(), 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue})
                        if args.save_cca_attn: #len(vqa_out) == 5:
                            cca_attns = vqa_out[4]
                            pt_attn = cca_attns['pt'].detach().cpu().numpy()
                            tp_attn = cca_attns['tp'].detach().cpu().numpy()
                            ps_attn = cca_attns['ps'].detach().cpu().numpy() if cca_attns['ps'] is not None else None
                            ts_attn = cca_attns['ts'].detach().cpu().numpy() if cca_attns['ts'] is not None else None
                            save_encoding_ques_ids = [21671000, 63525001, 341393003, 206922001, 443713002, 339705011, 349321003, 268539002, 563603000, 493724001, 288229001]
                            save_encoding = False
                            for qid in ques_id:
                                if qid in save_encoding_ques_ids:
                                    save_encoding = True
                                    break
                            if save_encoding:
                                write_attn(args.model_save_path+'_cca_attns_val_ma', {'pt_attn': pt_attn, 'tp_attn': tp_attn, 'ps_attn': ps_attn, 'ts_attn': ts_attn, 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue, 'temporal_cache': cca_attns['t_cache'].detach().cpu().numpy(), 'spatial_cache': cca_attns['sp_cache'].detach().cpu().numpy()})
                            else:
                                write_attn(args.model_save_path+'_cca_attns_val_ma', {'pt_attn': pt_attn, 'tp_attn': tp_attn, 'ps_attn': ps_attn, 'ts_attn': ts_attn, 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': if task == 'vqa_with_test':
                    test_loss = 0
                    test_acc=0
                    test_l1_loss_struct = 0
                    predictions = []
                    ans = []
                    ans_str = []
                    ques_ids = []
                    # print('-' * 100)
                    # print('Evaluation step')
                    log_str += '-'*100 + '\nTest step\n'
                    for b in tqdm(test_dl):
                        imgs = b['img']
                        answers=b['ans']
                        if len(b['struct']) != 0:
                            struct = b['struct'].long() if not args.one_hot_only else None
                            struct_one_hot = torch.as_tensor(one_hot_encoding(b['struct'].int().numpy(), num_cat_dict)).float() if not args.entity_only else None
                        else:
                            struct = None
                            struct_one_hot = None
                        # struct=b['struct'] if len(b['struct']) != 0 else None
                        if gpu_id >= 0:
                            imgs = imgs.cuda(device=gpu_id)
                            answers=answers.cuda(device=gpu_id)
                            if struct_one_hot is not None:
                                struct_one_hot = struct_one_hot.cuda(device=gpu_id)
                            if struct is not None:
                                struct = struct.cuda(device=gpu_id)
                        questions= b['ques']
                        # if hasattr(args,'two_ques_separate') and args.two_ques_separate:
                        #     questions = [q.split('? ') for q in b['ques']]
                        #     questions = [[q[0]+'?',q[1]] for q in questions]
                        # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                        vqa_out = r.vqa(model, imgs, questions,targets=answers,structured=struct,structured_one_hot=struct_one_hot,mode=args.eval_mode,return_str_preds=True, greedy_only=args.greedy_only)
                        pred, loss,acc,l1_loss_struct = vqa_out[:4]
                        test_loss += float(loss.detach().cpu().numpy())
                        if args.l1reg is not None:
                            struct_periph_l1_loss = sum([torch.sum(torch.abs(p)) for n,p in shared_model.named_parameters() if 'struct_periph' in n])
                            test_l1_loss_struct += args.l1reg*(float(struct_periph_l1_loss.detach().cpu().numpy()) + float(l1_loss_struct.detach().cpu().numpy()))
                        test_acc+=acc
                        ans += b['ans'].squeeze().tolist()
                        ans_str += b['ans_str']
                        ques_ids += b['ques_id']
                        predictions += pred.detach().cpu().squeeze().tolist()
                    test_loss/=len(val_dl)
                    test_acc=(val_acc/len(val_dl))
                    # summary_writer.add_scalar('Val_loss', val_loss, step)


                if task == 'vqa' or task == 'vqa_with_test':
                    non_ma_predictions = []
                    non_ma_ans = []
                    non_ma_ans_str = []
                    non_ma_ques_ids = []
                    for b in tqdm(val_non_ma_dl):
                        imgs = b['img']
                        answers=b['ans']
                        if len(b['struct']) != 0:
                            struct = b['struct'].long() if not args.one_hot_only else None
                            struct_one_hot = torch.as_tensor(one_hot_encoding(b['struct'].int().numpy(), num_cat_dict)).float() if not args.entity_only else None
                        else:
                            struct = None
                            struct_one_hot = None
                        # struct=b['struct'] if len(b['struct']) != 0 else None
                        if gpu_id >= 0:
                            imgs = imgs.cuda(device=gpu_id)
                            answers=answers.cuda(device=gpu_id)
                            if struct_one_hot is not None:
                                struct_one_hot = struct_one_hot.cuda(device=gpu_id)
                            if struct is not None:
                                struct = struct.cuda(device=gpu_id)
                        questions= b['ques']
                        # if hasattr(args,'two_ques_separate') and args.two_ques_separate:
                        #     questions = [q.split('? ') for q in b['ques']]
                        #     questions = [[q[0]+'?',q[1]] for q in questions]
                        # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                        vqa_out = r.vqa(model, imgs, questions,targets=answers,structured=struct,structured_one_hot=struct_one_hot,mode=args.eval_mode,return_str_preds=True, greedy_only=args.greedy_only)
                        pred, loss,acc,l1_loss_struct = vqa_out[:4]
                        if args.save_decoder_attn or args.save_cca_attn: #len(vqa_out) > 4:
                        # if len(vqa_out) > 4:
                            ques_id = b['ques_id']
                            vocab_file=os.path.join('conf/vqa_vocab.pkl')
                            with open(vocab_file,'rb') as f:
                                ans_to_id,id_to_ans=pickle.loads(f.read())
                            ytrue = torch.reshape(answers, [-1]).cpu().numpy()
                            preds = pred.detach().cpu().squeeze().tolist()
                            preds_str = [id_to_ans[pr] for pr in preds]
                            ytrue_str = [id_to_ans[yt] for yt in ytrue]

                            if args.save_decoder_attn: #len(vqa_out) == 6:
                            # if len(vqa_out) == 6:
                                dec_attns = vqa_out[5]
                                write_attn(args.model_save_path+'_dec_attns_val_non_ma', {'dec_spat_attn': dec_attns[1].detach().cpu().numpy(), 'dec_tmp_attn': dec_attns[2].detach().cpu().numpy(), 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue})
                            if args.save_cca_attn: #len(vqa_out) == 5:
                            # if len(vqa_out) == 5:
                                cca_attns = vqa_out[4]
                                pt_attn = cca_attns['pt'].detach().cpu().numpy()
                                tp_attn = cca_attns['tp'].detach().cpu().numpy()
                                ps_attn = cca_attns['ps'].detach().cpu().numpy() if cca_attns['ps'] is not None else None
                                ts_attn = cca_attns['ts'].detach().cpu().numpy() if cca_attns['ts'] is not None else None
                                write_attn(args.model_save_path+'_cca_attns_val_non_ma', {'pt_attn': pt_attn, 'tp_attn': tp_attn, 'ps_attn': ps_attn, 'ts_attn': ts_attn, 'ques_id': ques_id, 'preds_str': preds_str, 'preds': preds, 'ans_str': ytrue_str, 'ans': ytrue})
                        non_ma_predictions += pred.detach().cpu().squeeze().tolist()
                        non_ma_ans += b['ans'].squeeze().tolist()
                        non_ma_ans_str += b['ans_str']
                        non_ma_ques_ids += b['ques_id']
                    
                    all_predictions = predictions + non_ma_predictions
                    all_ans = ans + non_ma_ans
                    all_ans_str = ans_str + non_ma_ans_str
                    all_ques_ids = ques_ids + non_ma_ques_ids
                    all_simple_acc = 100*np.sum(np.array(all_predictions)==np.array(all_ans))/len(all_predictions)

                    result_json = []
                    for j in range(len(all_ques_ids)):
                        result_json.append({'question_id':all_ques_ids[j],'answer':id_to_ans[all_predictions[j]]})

                    if not os.path.exists(os.path.join(args.model_save_path,'res')):
                        os.makedirs(os.path.join(args.model_save_path,'res'))
                    with open(os.path.join(args.model_save_path, 'res/vqa_prediction.json'), 'w') as outfile:
                        json.dump(result_json, outfile)

                    vqa_predictions = []
                    vqaRes=vqa.loadRes(os.path.join(args.model_save_path, 'res/vqa_prediction.json'),vqa_val_ques)
                    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
                    with open(os.path.join(args.model_save_path, 'res/vqa_prediction.json'), 'r') as f:
                        json_ans = json.load(f)
                    for j in json_ans:
                        vqa_predictions.append(j['answer'])
                    vqa_predictions=np.array(vqa_predictions)
                    # print('vqa_eval_simple_accuracy:', np.sum(vqa_predictions==np.array(all_ans))/vqa_predictions.shape[0])
                    vqa_eval_simple_acc = 100*np.sum(vqa_predictions==np.array(all_ans_str))/vqa_predictions.shape[0]
                    # evaluate results
                    vqaEval.evaluate()
                    # print('Overall Accuracy is: %.02f\n' %(vqaEval.accuracy['overall']))
                    vqa_eval_overall_acc = vqaEval.accuracy['overall']

                    # all_simple_acc = 0
                    # vqa_eval_simple_acc = 0
                    # vqa_eval_overall_acc = 0
                else:
                    # vqa_metric
                    result_json = []
                    for j in range(len(ques_ids)):
                        result_json.append({'question_id':ques_ids[j],'answer':id_to_ans[predictions[j]]})

                    if not os.path.exists(os.path.join(args.model_save_path,'val_res')):
                        os.makedirs(os.path.join(args.model_save_path,'val_res'))
                    with open(os.path.join(args.model_save_path, 'val_res/vqa_prediction.json'), 'w') as outfile:
                        json.dump(result_json, outfile)

                    vqa_predictions = []
                    vqaRes=vqa_val.loadRes(os.path.join(args.model_save_path, 'val_res/vqa_prediction.json'),vqa_val_ques)
                    vqaEval = VQAEval(vqa_val, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
                    with open(os.path.join(args.model_save_path, 'val_res/vqa_prediction.json'), 'r') as f:
                        json_ans = json.load(f)
                    for j in json_ans:
                        vqa_predictions.append(j['answer'])
                    vqa_predictions=np.array(vqa_predictions)
                    # print('vqa_eval_simple_accuracy:', np.sum(vqa_predictions==np.array(all_ans))/vqa_predictions.shape[0])
                    vqa_eval_simple_acc = 100*np.sum(vqa_predictions==np.array(ans_str))/vqa_predictions.shape[0]
                    # evaluate results
                    vqaEval.evaluate(quesIds=ques_ids)
                    # print('Overall Accuracy is: %.02f\n' %(vqaEval.accuracy['overall']))
                    vqa_eval_overall_acc = vqaEval.accuracy['overall']

                    test_loss = 0
                    test_acc=0
                    test_l1_loss_struct = 0
                    predictions = []
                    ans = []
                    ans_str = []
                    ques_ids = []
                    # print('-' * 100)
                    # print('Evaluation step')
                    log_str += '-'*100 + '\nTest step\n'
                    for b in tqdm(test_dl):
                        imgs = b['img']
                        answers=b['ans']
                        if len(b['struct']) != 0:
                            struct = b['struct'].long() if not args.one_hot_only else None
                            struct_one_hot = torch.as_tensor(one_hot_encoding(b['struct'].int().numpy(), num_cat_dict)).float() if not args.entity_only else None
                        else:
                            struct = None
                            struct_one_hot = None
                        # struct=b['struct'] if len(b['struct']) != 0 else None
                        if gpu_id >= 0:
                            imgs = imgs.cuda(device=gpu_id)
                            answers=answers.cuda(device=gpu_id)
                            if struct_one_hot is not None:
                                struct_one_hot = struct_one_hot.cuda(device=gpu_id)
                            if struct is not None:
                                struct = struct.cuda(device=gpu_id)
                        questions= b['ques']
                        # if hasattr(args,'two_ques_separate') and args.two_ques_separate:
                        #     questions = [q.split('? ') for q in b['ques']]
                        #     questions = [[q[0]+'?',q[1]] for q in questions]
                        # In val mode we do not pass the targets for prediction. We use it only for loss calculation
                        vqa_out = r.vqa(model, imgs, questions,targets=answers,structured=struct,structured_one_hot=struct_one_hot,mode=args.eval_mode,return_str_preds=True, greedy_only=args.greedy_only)
                        pred, loss,acc,l1_loss_struct = vqa_out[:4]
                        test_loss += float(loss.detach().cpu().numpy())
                        if args.l1reg is not None:
                            struct_periph_l1_loss = sum([torch.sum(torch.abs(p)) for n,p in shared_model.named_parameters() if 'struct_periph' in n])
                            test_l1_loss_struct += args.l1reg*(float(struct_periph_l1_loss.detach().cpu().numpy()) + float(l1_loss_struct.detach().cpu().numpy()))
                        test_acc+=acc
                        ans += b['ans'].squeeze().tolist()
                        ans_str += b['ans_str']
                        ques_ids += b['ques_id']
                        predictions += pred.detach().cpu().squeeze().tolist()
                    test_loss/=len(test_dl)
                    test_acc=(test_acc/len(test_dl))
                    summary_writer.add_scalar('Test_loss', test_loss, step)

                    result_json = []
                    for j in range(len(ques_ids)):
                        result_json.append({'question_id':ques_ids[j],'answer':id_to_ans[predictions[j]]})

                    if not os.path.exists(os.path.join(args.model_save_path,'test_res')):
                        os.makedirs(os.path.join(args.model_save_path,'test_res'))
                    with open(os.path.join(args.model_save_path, 'test_res/vqa_prediction.json'), 'w') as outfile:
                        json.dump(result_json, outfile)

                    vqa_predictions = []
                    vqaRes=vqa_test.loadRes(os.path.join(args.model_save_path, 'test_res/vqa_prediction.json'),vqa_test_ques)
                    vqaEval = VQAEval(vqa_test, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
                    with open(os.path.join(args.model_save_path, 'test_res/vqa_prediction.json'), 'r') as f:
                        json_ans = json.load(f)
                    for j in json_ans:
                        vqa_predictions.append(j['answer'])
                    vqa_predictions=np.array(vqa_predictions)
                    # print('vqa_eval_simple_accuracy:', np.sum(vqa_predictions==np.array(all_ans))/vqa_predictions.shape[0])
                    vqa_eval_simple_acc_test = 100*np.sum(vqa_predictions==np.array(ans_str))/vqa_predictions.shape[0]
                    # evaluate results
                    vqaEval.evaluate(quesIds=ques_ids)
                    # print('Overall Accuracy is: %.02f\n' %(vqaEval.accuracy['overall']))
                    vqa_eval_overall_acc_test = vqaEval.accuracy['overall']

                if task == 'vqa_with_test':
                    log_str += 'Step %d, VQA validation loss: %f, Accuracy %f %%, test loss: %f, Accuracy %f %%, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%\n' % (step, val_loss,val_acc,test_loss,test_acc,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc)
                else:
                    if args.l1reg is None:
                        # print('Step %d, VQA validation loss: %f, Accuracy %f %%, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%' % (step, val_loss,val_acc,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc))
                        # print('-' * 100)
                        log_str += 'Step %d, VQA validation loss: %f, Accuracy %f %%, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%, vqa_eval_simple_acc_test %f %%, vqa_eval_overall_acc_test %f %%\n' % (step, val_loss,val_acc,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc,vqa_eval_simple_acc_test,vqa_eval_overall_acc_test)
                    else:
                        # print('Step %d, VQA validation loss: %f, Accuracy %f %%, L1 loss struct: %f, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%' % (step, val_loss,val_acc,val_l1_loss_struct,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc))
                        # print('-' * 100)
                        log_str += 'Step %d, VQA validation loss: %f, Accuracy %f %%, L1 loss struct: %f, all_simple_acc %f %%, vqa_eval_simple_acc %f %%, vqa_eval_overall_acc %f %%, vqa_eval_simple_acc_test %f %%, vqa_eval_overall_acc_test %f %%\n' % (step, val_loss,val_acc,val_l1_loss_struct,all_simple_acc,vqa_eval_simple_acc,vqa_eval_overall_acc,vqa_eval_simple_acc_test,vqa_eval_overall_acc_test)
                log_str += '-'*100 + '\n'

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_iteration = step-1
                    # print(best_iteration)
                    log_str += 'best_iteration:{}\n'.format(best_iteration)

                    shared_model.save(args.model_save_path, 'best/0')
                    optimizer.save(args.model_save_path, 'best/0')

                    with open(args.model_save_path + '/best/acc.pkl', 'wb') as f:
                        pickle.dump({'best_val_acc': best_val_acc, 'best_iteration': best_iteration}, f)

                print_log(log_str, args.model_save_path+'.log')
                log_str = ''

                model = model.train()
                continue
            # if args.tune_steps is not None and i % 50 == 0:
            #     print_log(log_str, args.model_save_path+'.log')
            #     log_str = ''if args.save_cca_attn:iters = [int(os.path.basename(c)) for c in ckpts if os.path.basename(c) not in ['best','res', 'val_res', 'test_res']]
if 'pp' in args.cca_streams:
            shared_model.restore_state_dict(load_vit_weights())
     