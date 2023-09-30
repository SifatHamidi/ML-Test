import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account


def get_google_sheet_data(data_range, service_account_file):
    """connect to google spreadsheet via Google sheet api to get data of provided sheet_id
    params:
          scopes: connect to spreadsheet api
          service_account_file: path of credential json file
          spreadsheet_id: From whom to get data
    """
    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes)

    # Here enter the id of google sheet
    spreadsheet_id = '1QQNAoPbNmSBbdACPXgYM-gnxGHipqFxsPupG8JStEO4'
    try:
        service = build('sheets', 'v4', credentials=credentials)

        # call the sheets api
        sheet = service.spreadsheets()
        api_response = sheet.values().get(spreadsheetId=spreadsheet_id,
                                          range=data_range).execute()

        # data processing
        spreadsheet_data = api_response.get('values', [])
        processed_dataframe = pd.DataFrame(spreadsheet_data[1:], columns=spreadsheet_data[0])
        print(processed_dataframe)

    except HttpError as error:
        print(error)


# Call the function
if __name__ == '__main__':
    get_google_sheet_data('A1:E4', 'credentials.json')
