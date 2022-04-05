from modules import*
from custom_dataset import*

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
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)


    # dataset_style = CustomImageDataset('.\\style_images\\test_samples',transform = transforms.Resize((512, 640)))
    # dataset_content = CustomImageDataset('.\\content_images\\test_samples',transform = transforms.Resize((512, 640)))
    transform = transforms.Compose([transforms.Resize((512, 640)), transforms.ToTensor()])

    dataset_style = CustomImageDataset('content_images/test_samples',transform)
    dataset_content = CustomImageDataset('style_images/test_samples',transform)


    batch_size = 10
    dataloader_style = DataLoader(dataset_style, batch_size=batch_size, shuffle=True)
    dataloader_content = DataLoader(dataset_content, batch_size=batch_size, shuffle=True)

    n_epochs = 2
    for epoch in range(n_epochs):

        #style_img = TF.resize(TF.to_tensor(style_img), (512, 640))  # TODO: Automatically find the needed size based on VGG-blocks used
        #content_img = TF.resize(TF.to_tensor(content_img), (512, 640))  # TODO: Automatically find the needed size based on VGG-blocks used

        #style_img.unsqueeze_(dim=0)
        #content_img.unsqueeze_(dim=0)
        for i, batch in enumerate(zip(dataloader_style,dataloader_content)):

            batch_style = batch[0]
            batch_content = batch[1]

            batch_style = batch_style.to(device)
            batch_content = batch_content.to(device)
            print(batch_style.shape)
            
            encoder_out_style = encoder(batch_style)      # TODO: Pass both images through encoder in one forward pass
            encoder_out_content = encoder(batch_content)
            adain_out = adain(encoder_out_content, encoder_out_style)
            decoder_out = decoder(adain_out)
            loss = style_loss(decoder_out, batch_style) + image_loss(decoder_out, batch_content)
            print(f"Epoch {epoch}, Batch {i}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch %5 == 0: ##Endre på hvor ofte du ønsker å lagre bildet
            img_tensor = decoder_out.detach().cpu().squeeze(dim=0)
            fig, axs = plt.subplots(batch_size,1,figsize=(5, 5*batch_size))
            for i in range(batch_size):
                img = img_tensor[i].permute(1,2,0).squeeze()
                axs[i].axis("off")
                axs[i].imshow(img)
            plt.savefig(f"progressimages/epoch{epoch}.png")