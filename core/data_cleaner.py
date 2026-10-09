import pandas as pd

from core.utils import clean_amount, clean_date, clean_text, extract_reference


def clean_sheet(dataframe):
    dataframe = dataframe.dropna(how="all").reset_index(drop=True)

    header_row = 0

    for index in range(min(20, len(dataframe))):
        row_text = " ".join(
            str(value).lower()
            for value in dataframe.iloc[index].values
        )

        if "date" in row_text and ("debit" in row_text or "credit" in row_text):
            header_row = index
            break

    dataframe = dataframe.iloc[header_row:]
    dataframe.columns = [str(column).strip() for column in dataframe.iloc[0]]
    dataframe = dataframe.iloc[1:].reset_index(drop=True)

    return dataframe


def find_column(dataframe, keywords):
    for column in dataframe.columns:
        column_name = str(column).lower()

        if any(keyword in column_name for keyword in keywords):
            return column

    return None


def standardize(dataframe, source_type, sheet_name):
    date_column = find_column(dataframe, ["date"])
    description_column = find_column(dataframe, ["description", "narration", "particulars"])
    debit_column = find_column(dataframe, ["debit", "withdrawal"])
    credit_column = find_column(dataframe, ["credit", "deposit"])

    if date_column is None or description_column is None:
        raise ValueError(f"{source_type} mein Date ya Description column nahi mila.")

    transactions = []

    for row_number, row in dataframe.iterrows():
        date = clean_date(row[date_column])
        description = clean_text(row[description_column])

        debit = clean_amount(row[debit_column]) if debit_column else 0.0
        credit = clean_amount(row[credit_column]) if credit_column else 0.0

        if date is None:
            continue

        if debit == 0 and credit == 0:
            continue

        if "opening balance" in description.lower():
            continue

        if source_type == "Bank Statement":
            direction = "Outflow" if debit > 0 else "Inflow"
            amount = debit if debit > 0 else credit
        else:
            direction = "Inflow" if debit > 0 else "Outflow"
            amount = debit if debit > 0 else credit

        transactions.append({
            "Source": source_type,
            "Sheet": sheet_name,
            "Row": row_number + 2,
            "Date": date,
            "Description": description,
            "Direction": direction,
            "Amount": amount,
            "Reference": extract_reference(description),
            "Status": "Unmatched",
            "Match ID": "",
            "Match Rule": "",
        })

    return pd.DataFrame(transactions)