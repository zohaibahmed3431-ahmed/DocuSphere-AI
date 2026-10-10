import pandas as pd

from core.utils import (
    clean_amount,
    clean_date,
    clean_text,
    extract_reference,
)


def clean_sheet(dataframe):
    dataframe = dataframe.dropna(how="all").reset_index(drop=True)

    header_row = 0

    for index in range(min(40, len(dataframe))):
        row_text = " ".join(
            str(value).lower()
            for value in dataframe.iloc[index].values
        )

        if "date" in row_text and (
            "debit" in row_text
            or "credit" in row_text
            or "amount" in row_text
        ):
            header_row = index
            break

    dataframe = dataframe.iloc[header_row:]
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.iloc[0].values
    ]
    dataframe = dataframe.iloc[1:].reset_index(drop=True)

    return dataframe


def find_column(dataframe, keywords):
    for column in dataframe.columns:
        column_name = str(column).lower()

        if any(keyword in column_name for keyword in keywords):
            return column

    return None


def standardize(dataframe, source_type, sheet_name, file_name):
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

    amount_column = find_column(
        dataframe,
        ["amount"],
    )

    if date_column is None:
        raise ValueError(
            f"{file_name} -> {sheet_name}: Date column nahi mila."
        )

    if description_column is None:
        raise ValueError(
            f"{file_name} -> {sheet_name}: Description column nahi mila."
        )

    transactions = []

    for row_number, row in dataframe.iterrows():
        date = clean_date(row[date_column])
        description = clean_text(row[description_column])

        debit = 0.0
        credit = 0.0
        amount = 0.0

        if debit_column is not None:
            debit = clean_amount(row[debit_column])

        if credit_column is not None:
            credit = clean_amount(row[credit_column])

        if amount_column is not None and debit == 0 and credit == 0:
            amount = clean_amount(row[amount_column])

        if date is None:
            continue

        if debit == 0 and credit == 0 and amount == 0:
            continue

        if "opening balance" in description.lower():
            continue

        if source_type == "Bank Statement":
            if debit > 0:
                direction = "Outflow"
                final_amount = debit
            elif credit > 0:
                direction = "Inflow"
                final_amount = credit
            else:
                direction = "Inflow"
                final_amount = amount

        else:
            if debit > 0:
                direction = "Inflow"
                final_amount = debit
            elif credit > 0:
                direction = "Outflow"
                final_amount = credit
            else:
                direction = "Outflow"
                final_amount = amount

        transactions.append({
            "Source": source_type,
            "File": file_name,
            "Sheet": sheet_name,
            "Row": row_number + 2,
            "Date": date,
            "Description": description,
            "Direction": direction,
            "Amount": round(final_amount, 2),
            "Reference": extract_reference(description),
            "Status": "Unmatched",
            "Match ID": "",
            "Match Rule": "",
            "Confidence": "",
        })

    return pd.DataFrame(transactions)