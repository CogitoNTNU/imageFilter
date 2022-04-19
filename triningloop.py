from modules import*
from custom_dataset import*
from copy import deepcopy
import numpy as np
import datetime
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(blocks=12)  # IDEAL BLOCKS ARE (depending on inclusion of 1 conv before output)5/6, 11/12, 19/20,
    adain = AdaIN()
    decoder = Decoder(blocks=12)

    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False


    encoder = encoder.to(device)
    adain = adain.to(device)
    decoder = decoder.to(device)


    style_loss = FeatureLoss(encoder, loss_weight=0.1)
    image_loss = ImageLoss(encoder.features[:6], pixel_loss_weight=2, feature_loss_weight=0.5)

    style_lamda = 1*10
    loss_criterion = AdaINLoss(style_lamda)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)


    # dataset_style = CustomImageDataset('.\\style_images\\test_samples',transform = transforms.Resize((512, 640)))
    # dataset_content = CustomImageDataset('.\\content_images\\test_samples',transform = transforms.Resize((512, 640)))
    transform = transforms.Compose([transforms.Resize((512, 640)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_style = CustomImageDataset('style_images_1/test_samples',transform)
    dataset_content = CustomImageDataset('content_images_1/test_samples',transform)


    batch_size = 1
    dataloader_style = DataLoader(dataset_content, batch_size=batch_size, shuffle=True)
    dataloader_content = DataLoader(dataset_style, batch_size=batch_size, shuffle=True)

    # Hooks for activations
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output #.clone().detach().cpu()
        return hook
    style_layers = [1,3,6,8,11]
    for i, layer in enumerate(style_layers):
        encoder.features[layer].register_forward_hook(get_activation(i))
    n_epochs = 10
    for epoch in range(n_epochs):

        #style_img = TF.resize(TF.to_tensor(style_img), (512, 640))  # TODO: Automatically find the needed size based on VGG-blocks used
        #content_img = TF.resize(TF.to_tensor(content_img), (512, 640))  # TODO: Automatically find the needed size based on VGG-blocks used

        #style_img.unsqueeze_(dim=0)
        #content_img.unsqueeze_(dim=0)
        for i, batch in enumerate(zip(dataloader_style,dataloader_content)):
            optimizer.zero_grad()
            batch_style = batch[0]
            batch_content = batch[1]

            batch_style = batch_style.to(device)
           
            batch_content = batch_content.to(device)
            
            encoder_out_style = encoder(batch_style)      # TODO: Pass both images through encoder in one forward pass
            style_activations = deepcopy(activation)
            encoder_out_content = encoder(batch_content)

          
            adain_out = adain(encoder_out_content, encoder_out_style)
            decoder_out = decoder(adain_out)
            encoder_out_output = encoder(decoder_out)
            output_activation = activation
        
            """
            Bad loss version 
            """
            # output embeding
            enc_out_o = encoder_out_output
            # Content embedding
            a_out = adain_out
            
            # style_activation
            style_a = [layer[0] for layer in style_activations.values()]
            print(style_a[0].shape)

            # output_activation
            output_activation = [layer[0] for layer in output_activation.values()]

            '''
            for b1 in zip(a_out, enc_out_o, *style_a, *output_activation):
                c_emb, out_emb, *rest = b1
                style_a = []
                for _ in style_layers:
                    a, *rest = rest
                    a.to(device)
                    style_a.append(a)
                output_a = []
                for _ in style_layers:
                    a, *rest = rest
                    a.to(device)
                    output_a.append(a)
                c_emb.to(device)
                out_emb.to(device)
                loss = loss_criterion.forward(c_emb, out_emb, style_a, output_a)
                loss.requires_grad = True
                loss.backward()
                optimizer.step()
                for el in style_a:
                    el.detach().cpu()
                for el in output_a:
                    el.detach().cpu()
           '''

            loss = loss_criterion.forward(a_out, enc_out_o, style_a, output_activation)
            loss.backward()
            optimizer.step()
            """
            Bad loss end
            """
            #loss = style_loss(decoder_out, batch_style) + image_loss(decoder_out, batch_content)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            print(f"Epoch {epoch}, Batch {i}: {loss.item()}")

        if epoch %1 == 0: ##Endre på hvor ofte du ønsker å lagre bildet
            img_tensor = decoder_out.detach().cpu()
            fig, axs = plt.subplots(batch_size,1,figsize=(5, 5*batch_size))
            for i in range(batch_size):
                print(img_tensor.shape)
                MEAN = torch.tensor([0.485, 0.456, 0.406])
                STD = torch.tensor([0.229, 0.224, 0.225])
                img = img_tensor[i]
                img = img * STD[:, None, None] + MEAN[:, None, None]
                img = img.permute(1,2,0).squeeze()
                #axs[i].axis("off")
                axs.imshow(img)
            plt.savefig(f"progressimages/epoch{epoch}.png")
    save_path = ".\\trained_model\\"+f'adain_model_{epoch}'
    torch.save(decoder, save_path)