import streamlit as st
from PIL import Image
import io
import pandas as pd

from pathlib import Path

from app import main as main_app

def main():
    st.title("Demo IA CHallenge Conaprole Equipo 1")

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
        output_dir = main_app(str("image.png"))

        processed_image_path = Path(output_dir).glob("*detection*").__next__()
        processed_image = Image.open(processed_image_path)

        df_path = Path(output_dir).glob("*.html").__next__()
        df = pd.read_html(df_path)[0]



        # Procesar la imagen (convertir a escala de grises)
        #processed_image = image.convert("L")

        # Mostrar la imagen procesada
        st.image(processed_image, caption='Imagen procesada).', use_column_width=True)

        # Mostrar dataframe
        st.write(df)

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
    main()