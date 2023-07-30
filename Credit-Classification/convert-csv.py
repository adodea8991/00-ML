import pandas as pd
from pyxlsb import open_workbook as open_xlsb

def convert_xlsb_to_csv(input_file, output_file):
    # Open the xlsb file
    with open_xlsb(input_file) as wb:
        # Select the first sheet
        with wb.get_sheet(1) as sheet:
            data = []
            for row in sheet.rows():
                data.append([item.v for item in row])
    
    # Convert data to DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    
    # Save DataFrame to CSV file
    df.to_csv(output_file, index=False)

# Convert test.xlsb to test.csv
convert_xlsb_to_csv("test.xlsb", "test.csv")

# Convert training.xlsb to training.csv
convert_xlsb_to_csv("training.xlsb", "training.csv")
