# FineTuning
A walkthrough of starting from scratch with Fine Tuning and what I learn.

I first started with fine tuning from a simple tutorial. 
**Starting with fine-tuning a ConvNet on image classification with ants and bees**

<img width="477" height="96" alt="Screenshot 2026-09-24 at 11 16 36 AM" src="https://github.com/user-attachments/assets/df53cf90-47c6-44bb-8c63-99f326a9d4fd" />

After running some fine-tuning experiments, I noticed that both training and validation accuracy fluctuated a lot between epochs. Was it the learning rate, or was the dataset too small? The Hymenoptera dataset has only 244 training images and 153 validation images, so I thought the model might be struggling to learn patterns that generalize. My batch size was also only 4 at first, which could have made training less stable.

<img width="186" height="293" alt="Screenshot 2026-09-24 at 11 17 38 AM" src="https://github.com/user-attachments/assets/da40ef0c-36fc-4708-baad-a6bc6724e864" />

Overall, accuracy was hovering around 50%—roughly as good as guessing between the two classes. When I increased the batch size from 4 to 16, things became slightly more stable, but accuracy still fluctuated around 50%. I also tried lowering the learning rate. That briefly brought training and validation accuracy up to around 90%, but both quickly fell again.

It turns out the problem was in my code: I had written zero_grad instead of calling zero_grad(). My gradients were accumulating across batches. So while I initially thought the small dataset or learning rate was causing the wild fluctuations, I needed to fix that bug first.

imageNetFineTuning.ipynb

**Can I fine-tune a model to classify medical images?** How accurate can it get?

<img width="328" height="255" alt="Screenshot 2026-09-24 at 11 17 55 AM" src="https://github.com/user-attachments/assets/03286dff-dcfd-4b90-baf6-3c20e4e87622" />

For this experiment, I used BioMedCLIP and the MedMNIST BloodMNIST dataset.
I started with closed-set zero-shot classification. Basically, I wanted to see how well BioMedCLIP could classify BloodMNIST’s eight cell types without training on those labels. I created a text prompt for each class, such as “a microscope image of a basophil,” and encoded those prompts alongside the images.
BioMedCLIP has two encoders: one for images and one for text. It learns to place matching image and text embeddings closer together, while pushing mismatched pairs farther apart. To get a score for each class, I compared the image embeddings with all eight text embeddings using image_features @ text_features.T.



Here’s the heat map from that experiment. The model almost never classified some cell types correctly, including basophils and eosinophils. Instead, it predicted a few other classes for many of them. Why? One possibility is that my text prompts didn’t capture the small visual differences between these cell types.
There could also be limitations in BioMedCLIP’s training data. For example, many images in PMC-15M are composite figures. Splitting those figures into individual panels might give the model more specific image–text pairs to learn from. BioMedCLIP also uses figure captions rather than all the surrounding text, though I’m less sure how much that matters here.
I tried shortening the prompts, but performance got worse. That made me think prompt wording was playing a role, although this experiment alone doesn’t tell me exactly why. In the end, zero-shot accuracy was 27%, and balanced accuracy was 20%.
Then I tried something else. What if the image embeddings already contained information about the cell types, but my text prompts just weren’t bringing it out? I kept BioMedCLIP’s image encoder and trained a single linear layer to map its 512-dimensional image embeddings to the eight classes. That worked much better:

<img width="446" height="322" alt="Screenshot 2026-09-24 at 11 19 50 AM" src="https://github.com/user-attachments/assets/1eefaa45-e86b-4f43-965b-27a3b31c6eef" />

This was exciting! It suggests that BioMedCLIP’s image embeddings did contain useful information about the cells. Even though the text prompts only got me 27% accuracy, a single trained linear layer could use those embeddings to reach 89% validation accuracy. I still need to check how well it does on a separate test set.

<img width="464" height="193" alt="Screenshot 2026-09-24 at 11 18 48 AM" src="https://github.com/user-attachments/assets/771a46fe-0c6f-48f8-b0aa-4120c6397056" />


And it did amazing!

