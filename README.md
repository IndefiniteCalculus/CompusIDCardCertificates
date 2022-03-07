# What does this system do
It's my classmate coorpurate project about OCR, face recognization and image processing, in order to identify the one and one's compus ID card is on the list or not. 

# How does this system works
The system would rotate and crop ID card's image loaded in, then identify the face area and characters area on the card, for further processing later on. 

The characters image area would be thresholding to enhance it's discernibility. The student list was stored on the remote database. 

The one's face image would be captured by real time camera to compare with the picture on ID card, make sure this ID card belongs to it's cardholder. 

# Future plan
The code I'm resposible for including OCR, image processing, backend access logic and GUI. These functions will be keep develop later on... maybe? 

I think the character recogization would work well by some **simple** and **light** machine learning ways, instead of **slow** and **fat** neural network. 

I used to spend lots of time implement a rubust character split method during the development, to test if some strategy can identify a **sigle characters** image croped from card. Not from all of the Chiese characters in the dictionary, just from the **set of character** in the student list. But something like KNN or PCA just don't work. 

Maybe I need some preprocessing or combination of some machine learning methods? 

By the way, this project is a demo for my course's final exam in 2020. Due to the remote database supported it had been offline, some function might be unavaliable in this show case for now.
