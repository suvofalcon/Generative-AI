import streamlit as st
from utils import create_docs, create_docs_llama2


def main():

    st.set_page_config(page_title="Invoice Extraction Bot")
    st.title("Invoice Extraction Bot...💁 ")
    st.subheader("I can help you in extracting invoice data")

    # Upload the invoices (pdf file)
    pdf = st.file_uploader("Upload invoices here, only PDF files allowed",
                           type=["pdf"],
                           accept_multiple_files=True)

    submit = st.button("Extract Data")

    if submit:
        with st.spinner("Wait for it ..."):
            # df = create_docs(pdf) - using OpenAI
            df = create_docs(pdf)  # using Llama2
            st.write(df.head())

            st.write("Now displaying contents from Llama2 response")
            response_list = create_docs_llama2(pdf)
            st.write(response_list)

            # save data as csv
            data_as_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download data as CSV",
                               data_as_csv,
                               "benchmark_tools.csv",
                               "text/csv",
                               key="download-tools-csv")

        st.success("Hope I was able to save your time...❤️")


# Invoking main function
if __name__ == "__main__":
    main()
