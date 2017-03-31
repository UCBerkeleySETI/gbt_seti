#include <mysql.h>

MYSQL * do_connect (char *host_name, char *user_name, char *password, char *db_name,
      unsigned int port_num, char *socket_name, unsigned int flags);

void do_disconnect (MYSQL **conn);
void exiterr(int exitcode);
void dbconnect(MYSQL **conn);

#define def_socket_name NULL /* use default socket name */

//MYSQL  *conn; 
/* pointer to connection handler */
MYSQL_RES *res;
MYSQL_ROW row;

char def_host_name[255];
char def_user_name[255];
char def_password[255];
int def_port_num;
char def_db_name[255];
