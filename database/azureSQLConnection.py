import pyodbc

def azureSQLServerDBConnection():
    azure_sql_connection = pyodbc.connect(
        server="tssgai-sqldbserver.database.windows.net",
        database="tssgai-db",
        user='raja.pellakuru',
        tds_version='7.4',
        password="T-Systems2019",
        port=1433,
        DRIVER='{ODBC Driver 17 for SQL Server}'
    )
    return azure_sql_connection

