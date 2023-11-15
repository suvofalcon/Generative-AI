import streamlit as st


# Define the streamlit app

def main():
    st.title("Customer Call Center Summarization")

    uploaded_files = st.file_uploader("Upload recorded .mp3 files", type=["mp3"],
                                      accept_multiple_files=True)

    if uploaded_files:
        st.write("Uploaded Files:")

        # Display the uploaded files and button in tabular form
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            col1, col2, col3 = st.columns([0.1, 1, 2])
            with col1:
                st.write("-")
            with col2:
                st.write(file_name)
            with col3:
                send_button = st.button(f"Send email for {file_name}")

                if send_button:
                    st.success(f"Sent email for {file_name}...!!")


# Run the streamlit app
if __name__ == "__main__":
    main()
