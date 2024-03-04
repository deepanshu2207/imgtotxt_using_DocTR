# import cv2
# import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import json

from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

from backend.pytorch import DET_ARCHS, RECO_ARCHS, load_predictor #forward_image

forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(det_archs, reco_archs):
    """Build a streamlit layout"""
    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("Document Text Extraction")
    # For newline
    st.write("\n")
    # Instructions
    st.markdown("*Hint: click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    # cols = st.columns((1, 1, 1, 1))
    cols = st.columns((1, 1, 1))
    cols[0].subheader("Input page")
    # cols[1].subheader("Segmentation heatmap")
    cols[1].subheader("OCR output")
    cols[2].subheader("Page reconstitution")

    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["pdf", "png", "jpeg", "jpg"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        page = doc[page_idx]
        cols[0].image(page)

    # Model selection
    st.sidebar.title("Model selection")
    det_arch = st.sidebar.selectbox("Text detection model", det_archs)
    reco_arch = st.sidebar.selectbox("Text recognition model", reco_archs)

    # # For newline
    # st.sidebar.write("\n")
    # # Only straight pages or possible rotation
    # st.sidebar.title("Parameters")
    # assume_straight_pages = st.sidebar.checkbox("Assume straight pages", value=True)
    # st.sidebar.write("\n")
    # # Straighten pages
    # straighten_pages = st.sidebar.checkbox("Straighten pages", value=False)
    # st.sidebar.write("\n")
    # # Binarization threshold
    # bin_thresh = st.sidebar.slider("Binarization threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    # st.sidebar.write("\n")

    if st.sidebar.button("Analyze page"):
        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner("Loading model..."):
                # Default Values
                assume_straight_pages, straighten_pages, bin_thresh = True, False, 0.3

                predictor = load_predictor(
                    det_arch, reco_arch, assume_straight_pages, straighten_pages, bin_thresh, forward_device
                )

            with st.spinner("Analyzing..."):
                # # Forward the image to the model
                # seg_map = forward_image(predictor, page, forward_device)
                # seg_map = np.squeeze(seg_map)
                # seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)

                # # Plot the raw heatmap
                # fig, ax = plt.subplots()
                # ax.imshow(seg_map)
                # ax.axis("off")
                # cols[1].pyplot(fig)

                # Plot OCR output
                out = predictor([page])
                fig = visualize_page(out.pages[0].export(), out.pages[0].page, interactive=False, add_labels=False)
                cols[1].pyplot(fig)

                # Page reconsitution under input page
                page_export = out.pages[0].export()
                if assume_straight_pages or (not assume_straight_pages and straighten_pages):
                    img = out.pages[0].synthesize()
                    cols[2].image(img, clamp=True)

                print('out',out)
                print('\n')
                print('page_export',page_export)
                print('\n')
                all_text = ''
                for i in page_export['blocks']:
                    for line in i['lines']:
                        for word in line['words']:
                            all_text+=word['value']
                            all_text+=' '
                        all_text+='\n'    
                
                print('all_text', all_text)
                print('\n')

                # Display Text
                st.markdown("\n## **Here is your text:**")
                st.write(all_text)

                # Display JSON
                json_string = json.dumps(page_export)
                st.markdown("\n## **Here are your analysis results in JSON format:**")
                st.download_button(label="Download JSON", data=json_string, file_name='data.json', mime='application/json')
                st.json(page_export, expanded=False)


if __name__ == "__main__":
    main(DET_ARCHS, RECO_ARCHS)
