#Code for 1 epoch not to flood training scripts

def train_one_epoch(epoch_index, tb_writer, model, optim, loss_func, train_dataloader, metric_calculator, metric_calculator_binary,  device):
    running_loss = 0.
    last_loss = 0.

    running_accu = 0.
    running_dice = 0.
    #running_precision = 0.
    #running_recall = 0.

    vert_running_accu = 0.
    vert_running_dice = 0.

    sc_running_accu = 0.
    sc_running_dice = 0.

    ivd_running_accu = 0.
    ivd_running_dice = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    #swap train dataloader for dset
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optim.zero_grad()

        
        # Make predictions for this batch
        outputs = model(inputs)

        #print("outputs shape", outputs.shape)
        
        #input output label shape [8 batchsize, 19 classes, row, col]


        # Compute the loss and its gradients
        loss = loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optim.step()

        #General metrics
        accu, dice, prec, recall = metric_calculator(labels, outputs)

        #Vertebrae Metrics (0-9)
        vert_labels = labels[:, :9, :, :]
        vert_outputs = outputs[:, :9, :, :]
        vert_accu, vert_dice, vert_prec, vert_recall = metric_calculator(vert_labels, vert_outputs)

        #Spinal Canal Metrics (10)
        sc_labels = labels[:, 10, :, :].unsqueeze(1)
        sc_outputs = outputs[:, 10, :, :].unsqueeze(1)
        sc_accu, sc_dice, sc_prec, sc_specificity, sc_recall = metric_calculator_binary(sc_labels, sc_outputs)

        #IVD Metrics (11-19)
        ivd_labels = labels[:, -9:, :, :]
        ivd_outputs = outputs[:, -9:, :, :]
        ivd_accu, ivd_dice, ivd_prec, ivd_recall = metric_calculator(ivd_labels, ivd_outputs)


        # Gather data and report
        #General
        running_loss += loss.item()
        running_accu += accu
        running_dice += dice
        #running_dice += dice
        #running_precision += prec
        #running_recall += recall

        #Vertebrae
        vert_running_accu += vert_accu
        vert_running_dice += vert_dice

        #Spinal Canal
        sc_running_accu += sc_accu
        sc_running_dice += sc_dice

        #IVD
        ivd_running_accu += ivd_accu
        ivd_running_dice += ivd_dice

        print("batch" ,i )
        #every 100 batches
        if i % 100 == 99: 
            last_loss = running_loss / 100 # loss per batch
            last_accu = running_accu / 100
            last_dice = running_dice / 100
            #last_precision = running_precision / 100
            #last_recall = running_recall / 100

            vert_last_accu = vert_running_accu / 100
            vert_last_dice = vert_running_dice /100

            sc_last_accu = sc_running_accu / 100
            sc_last_dice = sc_running_dice / 100

            ivd_last_accu = ivd_running_accu / 100
            ivd_last_dice = ivd_running_dice / 100

            #print('  batch {} loss: {}'.format(i + 1, last_loss))
            
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            
            tb_writer.add_scalar('General/accuracy_train', last_accu, tb_x)
            tb_writer.add_scalar('General/dice_train', last_dice, tb_x)

            tb_writer.add_scalar('Vertebrae/accuracy_train', vert_last_accu, tb_x)
            tb_writer.add_scalar('Vertebrae/dice_train', vert_last_dice, tb_x)

            tb_writer.add_scalar('Spinal Canal/accuracy_train', sc_last_accu, tb_x)
            tb_writer.add_scalar('Spinal Canal/dice_train', sc_last_dice, tb_x)

            tb_writer.add_scalar('Intervertebral Discs/accuracy_train', ivd_last_accu, tb_x)
            tb_writer.add_scalar('Intervertebral Discs/dice_train', ivd_last_dice, tb_x)
            
            running_loss = 0.
            running_accu = 0.
            running_dice = 0.

            vert_running_accu = 0.
            vert_running_dice = 0.

            
            sc_running_accu = 0.
            sc_running_dice = 0.

            
            ivd_running_accu = 0.
            ivd_running_dice = 0.

    #print("loss", loss)
    return last_loss