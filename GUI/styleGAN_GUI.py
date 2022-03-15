import PySimpleGUI as sg
from PIL import Image
import io
import os

folderPath = os.path.abspath(os.getcwd())
filePath_picture = folderPath + "/blue.jpg"
filePath_transfered = folderPath + "/blue.jpg"

titleField = [[sg.Text('Velkommen til styleGAN', size=(80, 1), justification='center', font=(25), relief=sg.RELIEF_RIDGE)]]

leftCol = [
            [sg.Image(key = 'IMAGE',filename=filePath_picture)],
            [sg.Text('Velg bilde:'),sg.InputText("Bilde",enable_events=True, key='PICTUREPATH',size=(20,1)) ,sg.FileBrowse()],
            [sg.HSeparator()],

            [sg.Image(key = 'STYLE',filename=filePath_picture)],
            [sg.Text('Velg stil:'),sg.InputText("Bilde",enable_events=True, key='STYLEPATH',size=(20,1)) ,sg.FileBrowse()],
            [sg.HSeparator()],

            [sg.Button("Overfør stil!",key='TRANSFER')]]

transferedCol = [[sg.Text("transfered column")],
                [sg.Image(key='TRANSPIC',filename=filePath_transfered)]]



layout= [titleField,[sg.Column(leftCol),sg.VSeparator(),sg.Column(transferedCol)]]

win = sg.Window('StyleganGUI',layout)


while True:
    event, values = win.read()

    if event == sg.WIN_CLOSED or event == '-EXIT-':
       break

    if event == 'PICTUREPATH':
        picturePath = values['PICTUREPATH']
        if os.path.exists(picturePath):
            image = Image.open(picturePath)
            image.thumbnail((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            win["IMAGE"].update(data=bio.getvalue())
win.close()
    