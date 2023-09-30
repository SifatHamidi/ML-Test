import psycopg2
from psycopg2 import extras
from psycopg2 import sql
from datetime import datetime

host_name = 'localhost'
database = 'product_management'
user = 'shifat'
db_password = 'master'
port_id = 5432


def connect_database(host_name, database, user, db_password, port_id):
    conn = None
    cursor_obj = None
    try:
        conn = psycopg2.connect(
            host=host_name,
            dbname=database,
            user=user,
            password=db_password,
            port=port_id)
        cursor_obj = conn.cursor(cursor_factory=extras.DictCursor)
        return conn, cursor_obj
    except Exception as e:
        return 400, f'Error occur during database connection{str(e)}'


def create_database(database_name):
    conn, cursor_obj = connect_database(host_name, database, user, db_password, port_id)
    if conn != 400:
        conn.autocommit = True
        database_query = """CREATE database %s""" % (database_name,)
        cursor_obj.execute(database_query)

        cursor_obj.close()
        conn.close()
        return f"Database:{database_name} has been created"
    else:
        return cursor_obj


def create_database_table():
    """
        Create table for specific database
        process:
                connecting to database
                create table using query
    """
    conn, cursor_obj = connect_database(host_name, database, user, db_password, port_id)
    try:
        create_tabel_script = """
            CREATE TABLE IF NOT EXISTS product (
                id SERIAL PRIMARY KEY,
                product_name varchar(40) NOT NULL,
                product_type varchar NOT NULL,
                price float,
                available_quantity int,
                sold_quantity int,
                stock_location_id int,
                stock_location_name varchar(20),
                entry_date timestamp without time zone DEFAULT (now() at time zone 'utc')
            ) """
        create_location_script = """
                    CREATE TABLE IF NOT EXISTS stock_location (
                        id SERIAL PRIMARY KEY,
                        location_name varchar(40) NOT NULL,
                        location_id int,
                        product_category varchar NOT NULL,
                        slot_occupied int,
                        slot_available int,
                        last_entry_date timestamp without time zone DEFAULT (now() at time zone 'utc')
                    ) """
        cursor_obj.execute(create_tabel_script)
        conn.commit()
        cursor_obj.close()
        return 'Successfully table created'
    except Exception as e:
        return f'Error while creating table for database product_management{str(e)}'

    finally:
        if conn is not None:
            conn.close()


def insert_data(values, table_name):
    """
        Create table for specific database
        process:
                connecting to database
                Add Data to specific table
        params:
              columns: list of mentioned table's column list
              values: data to be inserted
    """
    conn, cursor_obj = connect_database(host_name, database, user, db_password, port_id)
    try:
        insert_query = False

        if table_name == "product":
            columns = ['product_name', 'product_type', 'price', 'available_quantity', 'sold_quantity',
                       'stock_location_id', 'stock_location_name']
            header = ",".join(f'{column}' for column in columns)
            insert_query = ('INSERT INTO product({}) '
                            'VALUES(%s,%s,%s,%s,%s,%s,%s)').format(header)

        elif table_name == "location":
            columns = ['location_name', 'location_id', 'product_category', 'slot_occupied', 'slot_available']
            header = ",".join(f'{column}' for column in columns)
            insert_query = ('INSERT INTO stock_location({}) '
                            'VALUES(%s,%s,%s,%s,%s)').format(header)

        for data in values:
            cursor_obj.execute(insert_query, data)
            conn.commit()
        return 'Successfully data inserted into table'
    except Exception as e:
        return f'Error while inserting data into table for product_management database {str(e)}'
    finally:
        if conn is not None:
            cursor_obj.close()
            conn.close()


def fetch_data(table_name):
    """
        fetch data for specific database
        process:
                connecting to database
                fetch data to specific table
    """
    conn, cursor_obj = connect_database(host_name, database, user, db_password, port_id)
    try:
        fetch_query = False
        if table_name == 'product':
            fetch_query = """SELECT * FROM product 
                             WHERE product_name='100_Days_on_Moon' 
                             AND stock_location_id = 1"""
        elif table_name == 'stock_location':
            fetch_query = """SELECT * FROM stock_location 
                             WHERE location_name='Location A' """
        else:
            return 'Please provide valid table name for fetching data'

        cursor_obj.execute(fetch_query)
        query_output = cursor_obj.fetchall()

        return query_output
    except Exception as e:
        return f'Error while fetching data into table for product_management database {str(e)}'
    finally:
        if conn is not None:
            cursor_obj.close()
            conn.close()


def update_data(table_name, column_data):
    """
        update data for specific database
        process:
                connecting to database
                update data to given table_name
        params:
              table_name: table name  depending on which query will occur
              column_data: column value depending on which query will occur
    """
    conn, cursor_obj = connect_database(host_name, database, user, db_password, port_id)
    try:
        update_query = False
        if table_name == "product":
            update_query = f"""UPDATE product 
                              SET price = price + 200
                              WHERE product_name= \'{column_data}\'"""
        elif table_name == "stock_location":
            update_query = f"""UPDATE stock_location 
                              SET slot_available = slot_available + 10
                              WHERE location_name= \'{column_data}\' """
        else:
            return 'Please provide valid table name for updating data'

        cursor_obj.execute(update_query)
        conn.commit()

        return 'Successfully updated data'
    except Exception as e:
        return f'Error while updating data into table for product_management database {str(e)}'
    finally:
        if conn is not None:
            cursor_obj.close()
            conn.close()


def delete_data(table_name, row_name):
    """
        delete data for specific database
        process:
                connecting to database
                delete data to given table_name and other info
                pass the fetch data to ensure the delete process
    """
    conn, cursor_obj = connect_database(host_name, database, user, db_password, port_id)
    try:
        delete_query = False
        if table_name == 'product':
            delete_query = f"""DELETE FROM {table_name} 
                              WHERE product_name= \'{row_name}\'"""
        elif table_name == 'stock_location':
            delete_query = f"""DELETE FROM {table_name} 
                              WHERE location_name= \'{row_name}\'"""
        else:
            return 'Please provide valid table name for deleting data'

        cursor_obj.execute(delete_query)
        conn.commit()

        cursor_obj.execute(f"""SELECT * FROM {table_name}; """)
        query_output = cursor_obj.fetchall()

        return f'Successfully deleted data and the final table datas are {query_output}'
    except Exception as e:
        return f'Error while deleting data into table for product_management database {str(e)}'
    finally:
        if conn is not None:
            cursor_obj.close()
            conn.close()


"""creating database"""
database_creation_response = create_database('product_management')
print(database_creation_response)

"""creating database table"""
response = create_database_table()
print(response)

"""inserting data"""
product_values = [('Sofa', 'furniture', 150, 50, 10, 1, 'location D')]
location_values = [('Location A', 1, 'books,electronics', 10, 20,),
                   ('Location B', 2, 'books,electronics', 10, 22,),
                   ('Location D', 3, 'furniture', 8, 33,)]
insert_response = insert_data(location_values, 'location')
print(insert_response)

"""fetching data"""
fetch_response = fetch_data(table_name)
print(fetch_response)

"""updating data"""
update_response = update_data(table_name, column_data)
print(update_response)

"""deleting data"""
delete_response = delete_data('product', 'Sofa')
print(delete_response)
