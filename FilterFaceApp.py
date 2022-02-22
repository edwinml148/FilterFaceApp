

import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import numpy as np
import av

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

class VideoDeteccionCaraOjo:

    def __init__(self):
        self.threshold1 = 100

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6 , minSize=(self.threshold1,self.threshold1), maxSize=(self.threshold1+100,self.threshold1+100))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            gray_cara = cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
            # Para la deteccion de los ojos se mantiene la proporcion entre la cara y los ojos
            max_face = int(max(w, h)/4)
            eyes = eyes_detector.detectMultiScale(gray_cara, scaleFactor=1.1, minNeighbors=6 , minSize=(max_face,max_face), maxSize=(max_face+10,max_face+10))
            # Solo cuando se detecte dos ojos se dibujara.
            if ( len(eyes) == 2 ):
                for (x_e, y_e, w_e, h_e) in eyes:
                    cv2.rectangle(img[y:y+h,x:x+w], (x_e,y_e), (x_e+w_e,y_e+h_e), (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

class VideoBlurring:

    def __init__(self):
        self.threshold1 = 100
        self.kernel = 5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6 , minSize=(self.threshold1,self.threshold1), maxSize=(self.threshold1+100,self.threshold1+100))
        print(faces)
        for (x, y, w, h) in faces:
            kernel = np.ones((self.kernel,self.kernel),np.float32)/(self.kernel*self.kernel)
            img[y:y+h,x:x+w] = cv2.filter2D(img[y:y+h,x:x+w],-1,kernel)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class VideoModificacionFace:
    
    def __init__(self):
        self.imagen = None
        self.threshold1 = 100

    def recv(self, frame):
        
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6 , minSize=(self.threshold1,self.threshold1), maxSize=(self.threshold1+100,self.threshold1+100))


        for (x, y, w, h) in faces:
            b = cv2.resize( self.imagen , dsize=(h,w), interpolation=cv2.INTER_CUBIC)
            img[y:y+h,x:x+w] = b
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():

    with open('style.css') as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


    st.title('Crea tu propio filtro personalizado !!!')
    st.write('FilterFaceApp es un WebApp con el objetivo de practicar tus conocimientos de procesamiento de imagenes creando tus propios filtros sobre **rostros** e incorporarlo a la WebApp.')
    st.sidebar.header('Filtros personalizados')
    option = st.sidebar.selectbox('Elige una opcion ...',('Deteccion de caras y ojos', 'Blurring', 'Reemplazar cara por imagen'))
    
    if ( option ==  'Deteccion de caras y ojos'):
        
        #ctx = webrtc_streamer(key="example", video_transformer_factory=VideoDeteccionCaraOjo)
        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoDeteccionCaraOjo,
            rtc_configuration={ # Add this line
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        #ctx = webrtc_streamer(key="example", video_processor_factory=VideoDeteccionCaraOjo)

        if ctx.video_processor:
            ctx.video_processor.threshold1 = st.slider("Minimo tamaño de la cara", 100, 200, 150)

        st.write('*Minimo tamaño de la cara* : Indica la longitud en pixels minima de la cara detectada , a menor valor se podra detectar rostros mas lejos de la camara')
    
    if ( option ==  'Blurring'):
        #ctx = webrtc_streamer(key="example", video_transformer_factory=VideoBlurring)
        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoBlurring,
            rtc_configuration={ # Add this line
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        #ctx = webrtc_streamer(key="example", video_processor_factory=VideoBlurring)

        if ctx.video_processor:
            ctx.video_processor.threshold1 = st.slider("Minimo tamaño de la cara", 100, 200, 150)
            ctx.video_processor.kernel = st.slider("Grado de Blurring", 5, 50, 20)

        st.write('*Minimo tamaño de la cara* : Indica la longitud en pixels minima de la cara detectada , a menor valor se podra detectar rostros mas lejos de la camara')

    if ( option ==  'Reemplazar cara por imagen'):

        file_uploader = st.sidebar.file_uploader("Suba una imagen en formato jpg ...",type="jpg")
        
        if file_uploader is not None:

            file_bytes = np.asarray(bytearray(file_uploader.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            #ctx = webrtc_streamer(key="example", video_transformer_factory=VideoModificacionFace)
            #ctx = webrtc_streamer(key="example", video_processor_factory=VideoModificacionFace)
            ctx = webrtc_streamer(
                key="example",
                video_processor_factory=VideoModificacionFace,
                rtc_configuration={ # Add this line
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )

            

            if ctx.video_processor:
                ctx.video_processor.threshold1 = st.slider("Minimo tamaño de la cara", 100, 200, 150)
                ctx.video_processor.imagen = opencv_image

            st.write('*Minimo tamaño de la cara* : Indica la longitud en pixels minima de la cara detectada , a menor valor se podra detectar rostros mas lejos de la camara')
    

if __name__ == '__main__':
    main()