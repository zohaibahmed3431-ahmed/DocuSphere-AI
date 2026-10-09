import pandas as pd

from core.utils import (
    clean_amount,
    clean_date,
    clean_text,
    extract_reference,
)


def detect_header_row(dataframe):
    for index in range(min(20, len(dataframe))):
        row = dataframe.iloc[index]

        row_text = " ".join(
            clean_text(value).lower()
            for value in row.values
        )

        if any(
            keyword in row_text
            for keyword in [
                "date",
                "description",
                "debit",
                "credit",
                "amount",
            ]
        ):
            return index

    return 0


def clean_raw_sheet(dataframe):
    header_row = detect_header_row(dataframe)

    dataframe = dataframe.iloc[header_row:].reset_index(drop=True)

    dataframe.columns = [
        clean_text(column)
        for column in dataframe.iloc[0].values
    ]

    dataframe = dataframe.iloc[1:].reset_index(drop=True)

    dataframe = dataframe.dropna(how="all")

    return dataframe


def find_column(dataframe, hints):
    columns = {
        clean_text(column).lower(): column
        for column in dataframe.columns
    }

    for hint in hints:
        for column_name, original_column in columns.items():
            if hint in column_name:
                return original_column

    return None


def standardize_transactions(
    dataframe,
    source_type,
    sheet_name,
):
    date_column = find_column(
        dataframe,
        ["date", "transaction date", "posting date"],
    )

    description_column = find_column(
        dataframe,
        ["description", "narration", "particulars", "details"],
    )

    debit_column = find_column(
        dataframe,
        ["debit", "withdrawal", "money out"],
    )

    credit_column = find_column(
        dataframe,
        ["credit", "deposit", "money in"],
    )

    if date_column is None:
        raise ValueError(f"Date column not found in {sheet_name}.")

    if description_column is None:
        raise ValueError(f"Description column not found in {sheet_name}.")

    transactions = []

    for row_number, row in dataframe.iterrows():
        date = clean_date(row[date_column])
        description = clean_text(row[description_column])

        debit = 0.0
        credit = 0.0

        if debit_column is not None:
            debit = clean_amount(row[debit_column])

        if credit_column is not None:
            credit = clean_amount(row[credit_column])

        if date is None:
            continue

        if debit == 0 and credit == 0:
            continue

        if "opening balance" in description.lower():
            continue

        reference = extract_reference(description)

        if source_type == "Bank":
            direction = "Outflow" if debit > 0 else "Inflow"
            amount = debit if debit > 0 else credit
        else:
            direction = "Inflow" if debit > 0 else "Outflow"
            amount = debit if debit > 0 else credit

        transactions.append(
            {
                "Transaction ID": f"{source_type[:3].upper()}-{row_number + 1}",
                "Source": source_type,
                "Sheet": sheet_name,
                "Row": row_number + 2,
                "Date": date,
                "Description": description,
                "Direction": direction,
                "Amount": round(amount, 2),
                "Reference": reference,
                "Status": "Unmatched",
                "Match ID": "",
                "Matched With": "",
                "Match Rule": "",
                "Confidence": "",
            }
        )

    return pd.DataFrame(transactions)