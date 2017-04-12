#include <stdio.h>
#include <mysql.h>
#include "setimysql.h"
#include <stdlib.h>



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
do_disconnect (MYSQL **conn)
{
  mysql_close (*conn);
}

void exiterr(int exitcode)
{
	fprintf( stderr, "ERROR\n");
	exit( exitcode );
}

void dbconnect(MYSQL **conn) {

	char *envvar = NULL;

  strcpy(def_host_name, "104.154.94.28");
  strcpy(def_user_name, "obs");
  strcpy(def_password, "");
  def_port_num = 3306;
  strcpy(def_db_name, "nwfb");


    envvar = getenv("SETI_HOST_NAME");
    if (!envvar) {
        fprintf(stderr, "Missing environment variable: SETI_HOST_NAME\n");
		exit(1);
    }
	
	
    strcpy(def_host_name, envvar);

    envvar = getenv("SETI_DATABASE_NAME");
    if (!envvar) {
        fprintf(stderr, "Missing environment variable: SETI_DATABASE_NAME\n");
        exit(1);
    }

    strcpy(def_db_name, envvar);


    envvar = getenv("SETI_USER_NAME");
    if (!envvar) {
        fprintf(stderr, "Missing environment variable: SETI_USER_NAME\n");
        exit(1);
    }

    strcpy(def_user_name, envvar);

    envvar = getenv("SETI_PORT_NUM");
    if (!envvar) {
        fprintf(stderr, "Missing environment variable: SETI_DATABASE_NUM\n");
        exit(1);
    }
    def_port_num = atoi(envvar);



  printf("connecting...\n");


  *conn =  mysql_init(NULL); 


  if (mysql_real_connect (*conn, def_host_name, def_user_name, def_password,
            def_db_name, def_port_num, def_socket_name, 0) == NULL)
  {
    fprintf (stderr, "mysql_real_connect() failed:\nError %u (%s)\n",
              mysql_errno (conn), mysql_error (conn));
    return (NULL);
  }

	printf("mysql connected\n");

  if (*conn == NULL) {
	printf("mysql connect failed!\n");
  	exiterr(1);
  }

	/* Connect to database */    
	if (mysql_select_db(*conn,def_db_name))
		exiterr(2);
		
	printf("db connected\n");
		

}