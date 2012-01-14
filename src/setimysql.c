#include <stdio.h>
#include <mysql.h>
#include "setimysql.h"

MYSQL *
do_connect (char *host_name, char *user_name, char *password, char *db_name,
      unsigned int port_num, char *socket_name, unsigned int flags)
{
MYSQL  *conn; /* pointer to connection handler */

  conn = mysql_init (NULL);  /* allocate, initialize connection handler */
  if (conn == NULL)
  {
    fprintf (stderr, "mysql_init() failed\n");
    return (NULL);
  }
  if (mysql_real_connect (conn, host_name, user_name, password,
            db_name, port_num, socket_name, flags) == NULL)
  {
    fprintf (stderr, "mysql_real_connect() failed:\nError %u (%s)\n",
              mysql_errno (conn), mysql_error (conn));
    return (NULL);
  }
  return (conn);     /* connection is established */
}

void
do_disconnect (MYSQL *conn)
{
  mysql_close (conn);
}

void exiterr(int exitcode)
{
	fprintf( stderr, "%s\n", mysql_error(conn) );
	exit( exitcode );
}

void dbconnect() {


strcpy(def_host_name, "localhost");
strcpy(def_user_name, "root");
//strcpy(def_password, "foobar");
def_port_num = 9191;
strcpy(def_db_name, "keplerseti");

  printf("connecting...\n");
/* Open connection to SQL Server */
  conn = do_connect (def_host_name, def_user_name, def_password, def_db_name,
                  def_port_num, def_socket_name, 0);


  if (conn == NULL) {
	printf("mysql connect failed!\n");
  	exiterr(1);
  }

/* Connect to database */    
	if (mysql_select_db(conn,def_db_name))
		exiterr(2);

}