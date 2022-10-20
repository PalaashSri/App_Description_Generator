# App_Description_Generator

First deliverable of Honours Project: Website developed using Django that hosts automatic app description generation capability. The backend functionality is located at this python file appUpload/Backend/extract_apks.py. It contains the feature extraction code and sends the extracted tokens to the deep learning model to app description.

![Figure3](https://user-images.githubusercontent.com/52162785/196105046-f561ed1f-037f-4680-b71d-b9fa0c432fdd.PNG)

The website is currently only accessible on local host:  http://127.0.0.1:8000/appUpload/. 

Below are the steps required to install required packages, setup the project and open the website:

* Download model3 available on this link: https://drive.google.com/drive/folders/11YOhr31F830Ss1lqBgXV_kA506UGbTHU?usp=sharing. 
This is the deep learning text transformer model developed by Suyu Ma (co-supervisor of this project). Place this folder in the following location: appUpload/Backend/.

* Download glove.6B.100d.txt file available on this link: https://drive.google.com/drive/folders/11YOhr31F830Ss1lqBgXV_kA506UGbTHU?usp=sharing. Place this in the following location: appUpload/Backend/

* Run `pip install -U Celery` to install Celery on to the system.

* Install Rabbitmq by following the process mentioned in: https://www.rabbitmq.com/install-windows.html

* After rabbitmq has been setup properly, go to the location where rabbitmq has been installed and go to the sbin folder using the terminal. Example location on my desktop: "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.11.0\sbin". When in this folder, using the terminal execute the following command `.\rabbitmq-server.bat start`. This will start the rabbitmq servers.

* Then in the folder which contains this project run the following command `celery -A appExtractionSite worker -l info -E` to start celery worker. 

* Then run `python manage.py runserver` the following command to start the website server.  Visit the following link: http://127.0.0.1:8000/appUpload/ to view the website.

<b> Note: </b> You might be required to generate a Django secret key. Any Django secret key generator from the web can be used for this purpose. 
