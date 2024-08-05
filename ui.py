import streamlit as st
from PIL import Image
import io
import pandas as pd

from pathlib import Path

from app import main as main_app

def main(model_path="weights/best.pt"):
    st.title("Demo IA CHallenge Conaprole Equipo 1")


    conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.60, 0.05)
    iou_thres = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
    #check box to select if you want to split the image in patches
    split_image = st.checkbox("Split image in patches", value=False)

    # Subir imagen
    uploaded_file = st.file_uploader("Seleccionar una imagen...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        print(uploaded_file)
        # Leer la imagen
        #get the image path from uploaded file

        image = Image.open(uploaded_file)
        #save the image to a file
        image.save("image.png")

        st.image(image, caption='Imagen.', use_column_width=True)
        st.write("")

        st.write("Procesando imagen...")
        output_dir = main_app(str("image.png"), conf_th=conf_thres, iou_th=iou_thres, model_path=model_path, split_image= split_image)

        processed_image_path = Path(output_dir).glob("*full_image*").__next__()
        processed_image = Image.open(processed_image_path)

        processed_image_conaprole_path = Path(output_dir).glob("*full_image_conaprole*").__next__()
        processed_image_conaprole = Image.open(processed_image_conaprole_path)

        df_path = Path(output_dir).glob("*metrics.html").__next__()
        df_details = pd.read_html(df_path, index_col=0)[0]

        df_path = Path(output_dir).glob("*global.html").__next__()
        df_global = pd.read_html(df_path, index_col=0)[0]

        # Procesar la imagen (convertir a escala de grises)
        #processed_image = image.convert("L")

        # Mostrar la imagen procesada
        st.image(processed_image, caption='Imagen procesada con delineamiento de productos. Primera Etapa', use_column_width=True)
        st.image(processed_image_conaprole, caption='Imagen procesada con identificacion de productos. Segunda Etapa', use_column_width=True)

        # Mostrar dataframe
        st.write(df_global, "Share of Conaprole")
        st.write(df_details, "Detalles de la detección")

        # Descargar la imagen procesada
        buf = io.BytesIO()
        processed_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Descargar imagen procesada",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png",
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="weights/best.pt")
    args = parser.parse_args()

    main(args.model_path)