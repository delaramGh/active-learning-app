import os
import json
import pandas as pd
import streamlit as st
from PIL import Image
import active_learning_v3 as dgh
from active_learning_v3 import STATUS  

from streamlit_modal import Modal


# Constants
IMAGE_DIR = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\Delaram_work\\active learning\\dataset\\high_fog"
ORIGINAL_IMAGE_DIR = "C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\Delaram_work\\active learning\\dataset\\original_dataset"
STATE_FILE = "state.json"

LABELS = ["Ok", "Lost"]
labels_dict = {
    "Ok": {
        "emoji": "✅",
        "button_text": "OK",
        "help": "The modified image is a realistic representation of the original image \
            transformed in accordance with the chosen transformation.",
    },
    "Lost": {
        "emoji": "🚫",
        "button_text": "Loss",
        "help": "Objects in the modified image appear distorted, blurred, \
            or significantly altered, rendering them unrecognizable.",
    },
}



def initialize_state(state):
    state["selected_index"] = 0
    state["pre_processing"] = 0
    state["sub_dataset"]    = 0


def load_state(state):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            loaded_state = json.load(f)
            state["selected_index"] = loaded_state["selected_index"]
            state["pre_processing"] = loaded_state["pre_processing"]
            state["sub_dataset"]    = loaded_state["sub_dataset"]
    else:
        initialize_state(state) #selected index = 0


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump({"selected_index": state["selected_index"],
                   "pre_processing": state["pre_processing"],
                   "sub_dataset"   : state["sub_dataset"],}, f)


def display_images(selected_index, image_files):
    # Display the images
    image = Image.open(f"{IMAGE_DIR}/{image_files[selected_index]}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Modified Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Original Image")
        # Check if corresponding image in ORIGINAL_IMAGE_DIR exists and display it
        try:
            original_image_name = image_files[selected_index].replace("_torchvision", "")
            original_image = Image.open(f"{ORIGINAL_IMAGE_DIR}/{original_image_name}")
            st.image(original_image, use_column_width=True)
        except FileNotFoundError:
            st.error("No original image found.")


def prev_image(state):
    if state["selected_index"] > 0:
        state["selected_index"] -= 1


def next_image(state, image_files):
    if state["selected_index"] < len(image_files) - 1:
        state["selected_index"] += 1


def write_states(STATUS):
    # for i in STATUS.ARR.values():
    #     if i != "":
    #         st.write(i)
    st.write("- Model trainig is done!")
    st.write("- 270 (56 new) data are labeled automatically. ")
    st.write("- 330 data are unlabeled. ")
    st.write("   ")
    st.write("   ")



######################################################################
######################################################################
        
def main():
    # Session state
    if not st.session_state:
        print("main: load session states")
        load_state(st.session_state)
    state = st.session_state

    if state["pre_processing"] == 0:
        #change the state 
        save_state(state)
        STATUS.ARR["create_csv_file"] = dgh.create_csv_file(ORIGINAL_IMAGE_DIR, IMAGE_DIR)
        STATUS.ARR["preprocessing2"] = dgh.preprocessing2(ORIGINAL_IMAGE_DIR, IMAGE_DIR)
        state["pre_processing"] = 1

    #load image files
    if not os.path.exists(dgh.sub_csv_path__):
        if state["sub_dataset"] == 0:
            state["sub_dataset"] = 1
        STATUS.ARR["create_sub_dataset"] = dgh.create_sub_dataset(state)

    image_files = dgh.get_image_files()
    # Load labels from CSV file
    df_labels = dgh.load_labels()

    st.set_page_config(layout="wide")
    st.title("Active Image Labeling App")
        

    # Show Statistics    
    modal = Modal(key="Demo Key",title="Active-Learning Statistics 📈📊")

    if st.sidebar.button("Show Statistics 📈📊"):
        with modal.container():
                write_states(STATUS)
                dgh.report()
    st.sidebar.markdown("""---""")
    st.sidebar.markdown(""" """)

    # Layout for navigation buttons and label selection
    col1, col2 = st.sidebar.columns([1, 1])
    if col1.button("⬅️  Prev Image", use_container_width=True):
        prev_image(state)
    if col2.button("Next Image  ➡️", use_container_width=True):
        next_image(state, image_files)


    st.sidebar.markdown("""---""")
    # Label buttons
    for label in LABELS:
        label_button = st.sidebar.button(
            label=f"{labels_dict[label]['button_text']} {labels_dict[label]['emoji']}",
            key=label,
            use_container_width=True,
        )  # Add your preferred emojis here
        if label == "Ok":
            st.sidebar.markdown(""" """)
            st.sidebar.markdown(""" """)
        
        if label_button: #button clicked
            save_state(state)
            dgh.update_labels(df_labels, image_files[state["selected_index"]], label)
            st.toast(f"Image {image_files[state['selected_index']]} was labeled as {label}")
            next_image(state, image_files)

    st.sidebar.markdown("""---""")
    # Save buttons
    if st.sidebar.button("💾 Save State", use_container_width=True):
        save_state(state)
        st.sidebar.success("State saved.")

    st.sidebar.markdown(""" """)
    st.sidebar.markdown(""" """)
    number = st.sidebar.number_input('Image Index', min_value=-1, 
                                    max_value=len(image_files), value=state["selected_index"]+1)
    if st.sidebar.button("Go to Image Index", use_container_width=True):
        state["selected_index"] = number - 1

    # Display image number
    st.write(f"Image {state['selected_index'] + 1} of {len(image_files)}")

    if dgh.check_sub_dataset_compeleted():
        st.balloons()
        STATUS.ARR["merge_datasets"] = dgh.merge_datasets()
        STATUS.ARR["model_train"], STATUS.ARR["predict"]= dgh.handle_training()
        state["sub_dataset"] = 2
        state["selected_index"] = -1
        if dgh.number_of_unlabeled_data(dgh.csv_path__) != 0:
            dgh.create_sub_dataset(state)
        else:
            st.success("All of the data is labeled! 🏆☑️")


    display_images(state["selected_index"], image_files)

    # Find the row for the current image
    row = df_labels[df_labels["generated"] == image_files[state["selected_index"]]]

    # If the row exists, display the previous label
    with st.expander("Previous Label and Predictions", expanded=True):
        if not row.empty:
            previous_label = row.iloc[0]["Label"]
            st.markdown(f"**Previous label:** {previous_label}")
    
    


if __name__ == "__main__":
    main()
