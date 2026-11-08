#include "my_application.h"
#include <cstdlib>

int main(int argc, char** argv) {
  // Prevent GTK3 crash on newer GNOME/Wayland versions due to missing xsettings schema
  setenv("GDK_BACKEND", "x11", 0);

  g_autoptr(MyApplication) app = my_application_new();
  return g_application_run(G_APPLICATION(app), argc, argv);
}

