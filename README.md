# FineTuning
A walkthrough of starting from scratch with Fine Tuning and what I learn.

I first started with fine tuning from a simple tutorial. 
**Starting with fine-tuning a ConvNet on image classification with ants and bees**<img width="469" height="192" alt="Screenshot 2026-09-24 at 11 18 34 AM" src="https://github.com/user-attachments/assets/3675717a-cc6b-4bfb-9367-a040252a596d" />


<img width="477" height="96" alt="Screenshot 2026-09-24 at 11 16 36 AM" src="https://github.com/user-attachments/assets/df53cf90-47c6-44bb-8c63-99f326a9d4fd" />

After running some fine-tuning experiments, I noticed that both training and validation accuracy fluctuated a lot between epochs. Was it the learning rate, or was the dataset too small? The Hymenoptera dataset has only 244 training images and 153 validation images, so I thought the model might be struggling to learn patterns that generalize. My batch size was also only 4 at first, which could have made training less stable.

<img width="186" height="293" alt="Screenshot 2026-09-24 at 11 17 38 AM" src="https://github.com/user-attachments/assets/da40ef0c-36fc-4708-baad-a6bc6724e864" />

Overall, accuracy was hovering around 50%—roughly as good as guessing between the two classes. When I increased the batch size from 4 to 16, things became slightly more stable, but accuracy still fluctuated around 50%. I also tried lowering the learning rate. That briefly brought training and validation accuracy up to around 90%, but both quickly fell again.

It turns out the problem was in my code: I had written zero_grad instead of calling zero_grad(). My gradients were accumulating across batches. So while I initially thought the small dataset or learning rate was causing the wild fluctuations, I needed to fix that bug first.

imageNetFineTuning.ipynb

**Can I fine-tune a model to classify medical images?** How accurate can it get?<img width="328" height="255" alt="Screenshot 2026-09-24 at 11 17 55 AM" src="https://github.com/user-attachments/assets/03286dff-dcfd-4b90-baf6-3c20e4e87622" />

For this experiment, I used BioMedCLIP and the MedMNIST BloodMNIST dataset.
I started with closed-set zero-shot classification. Basically, I wanted to see how well BioMedCLIP could classify BloodMNIST’s eight cell types without training on those labels. I created a text prompt for each class, such as “a microscope image of a basophil,” and encoded those prompts alongside the images.
BioMedCLIP has two encoders: one for images and one for text. It learns to place matching image and text embeddings closer together, while pushing mismatched pairs farther apart. To get a score for each class, I compared the image embeddings with all eight text embeddings using image_features @ text_features.T.



Here’s the heat map from that experiment. The model almost never classified some cell types correctly, including basophils and eosinophils. Instead, it predicted a few other classes for many of them. Why? One possibility is that my text prompts didn’t capture the small visual differences between these cell types.
There could also be limitations in BioMedCLIP’s training data. For example, many images in PMC-15M are composite figures. Splitting those figures into individual panels might give the model more specific image–text pairs to learn from. BioMedCLIP also uses figure captions rather than all the surrounding text, though I’m less sure how much that matters here.
I tried shortening the prompts, but performance got worse. That made me think prompt wording was playing a role, although this experiment alone doesn’t tell me exactly why. In the end, zero-shot accuracy was 27%, and balanced accuracy was 20%.
Then I tried something else. What if the image embeddings already contained information about the cell types, but my text prompts just weren’t bringing it out? I kept BioMedCLIP’s image encoder and trained a single linear layer to map its 512-dimensional image embeddings to the eight classes. That worked much better:

Epoch 01/20 | Train Loss: 1.9024 | Train Acc: 0.3443 | Val Loss: 1.7903 | Val Acc: 0.5987
Epoch 02/20 | Train Loss: 1.6947 | Train Acc: 0.6114 | Val Loss: 1.6017 | Val Acc: 0.6227
Epoch 03/20 | Train Loss: 1.5206 | Train Acc: 0.6514 | Val Loss: 1.4414 | Val Acc: 0.6694
Epoch 04/20 | Train Loss: 1.3737 | Train Acc: 0.6894 | Val Loss: 1.3055 | Val Acc: 0.7056
Epoch 05/20 | Train Loss: 1.2492 | Train Acc: 0.7239 | Val Loss: 1.1898 | Val Acc: 0.7278
Epoch 06/20 | Train Loss: 1.1437 | Train Acc: 0.7527 | Val Loss: 1.0919 | Val Acc: 0.7547
Epoch 07/20 | Train Loss: 1.0537 | Train Acc: 0.7813 | Val Loss: 1.0101 | Val Acc: 0.7985
Epoch 08/20 | Train Loss: 0.9771 | Train Acc: 0.8014 | Val Loss: 0.9393 | Val Acc: 0.8318
Epoch 09/20 | Train Loss: 0.9111 | Train Acc: 0.8221 | Val Loss: 0.8752 | Val Acc: 0.8271
Epoch 10/20 | Train Loss: 0.8537 | Train Acc: 0.8344 | Val Loss: 0.8215 | Val Acc: 0.8435
Epoch 11/20 | Train Loss: 0.8035 | Train Acc: 0.8439 | Val Loss: 0.7738 | Val Acc: 0.8569
Epoch 12/20 | Train Loss: 0.7590 | Train Acc: 0.8543 | Val Loss: 0.7339 | Val Acc: 0.8423
Epoch 13/20 | Train Loss: 0.7204 | Train Acc: 0.8567 | Val Loss: 0.6947 | Val Acc: 0.8692
Epoch 14/20 | Train Loss: 0.6854 | Train Acc: 0.8627 | Val Loss: 0.6611 | Val Acc: 0.8768
Epoch 15/20 | Train Loss: 0.6541 | Train Acc: 0.8674 | Val Loss: 0.6312 | Val Acc: 0.8768
Epoch 16/20 | Train Loss: 0.6259 | Train Acc: 0.8706 | Val Loss: 0.6045 | Val Acc: 0.8773
Epoch 17/20 | Train Loss: 0.6007 | Train Acc: 0.8738 | Val Loss: 0.5805 | Val Acc: 0.8890
Epoch 18/20 | Train Loss: 0.5775 | Train Acc: 0.8780 | Val Loss: 0.5587 | Val Acc: 0.8855
Epoch 19/20 | Train Loss: 0.5565 | Train Acc: 0.8798 | Val Loss: 0.5373 | Val Acc: 0.8855
Epoch 20/20 | Train Loss: 0.5373 | Train Acc: 0.8822 | Val Loss: 0.5185 | Val Acc: 0.8914

Best validation accuracy: 0.8914

This was exciting! It suggests that BioMedCLIP’s image embeddings did contain useful information about the cells. Even though the text prompts only got me 27% accuracy, a single trained linear layer could use those embeddings to reach 89% validation accuracy. I still need to check how well it does on a separate test set.

<img width="464" height="193" alt="Screenshot 2026-09-24 at 11 18 48 AM" src="https://github.com/user-attachments/assets/771a46fe-0c6f-48f8-b0aa-4120c6397056" />


And it did amazing!

