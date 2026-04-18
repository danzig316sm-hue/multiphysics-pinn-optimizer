

void setLibraryPath(const char *str);

bool createInstance(int clDevice, int groupSize);
void setOmpNumThreads(int num);
void loadProject(const char *str);
void loadSimDefinition(const char *str);
void initializeSimulation();

void advanceController_at_num(double *vars, int num);
bool advanceTurbineSimulation();
bool runFullSimulation();

void storeProject(const char *str);
void exportResults(int type, const char *filepath, const char* filename, const char *filter);
void closeInstance();
void setLogFile(const char *str);

void loadTurbulentWindBinary(const char *str);
void addTurbulentWind(double windspeed, double refheight, double hubheight, double dimensions, int gridPoints, double length, double dT, const char *turbulenceClass, const char *turbulenceType, int seed, double vertInf, double horInf, bool removeFiles);

void setPowerLawWind(double windspeed, double horAngle, double vertAngle, double shearExponent, double referenceHeight);
void setDebugInfo(bool isDebug);
void setUseOpenCl(bool isOpenCl);
void setAutoCleanup(bool enabled);

void setGranularDebug(bool dStr, bool dSim, bool dTurb, bool dCont, bool dSer);
void setTimestepSize(double timestep);
void setRPMPrescribeType_at_num(int type, int num);
void setRPM_at_num(double rpm, int num);
void setRampupTime(double time);
void setInitialConditions_at_num(double yaw, double pitch, double azimuth, double rpm, int num);
void setTurbinePosition_at_num(double x, double y, double z, double rotx, double roty, double rotz, int num);
void setControlVars_at_num(double *vars, int num);
void setExternalAction(const char *action, const char *id, double val, double pos, const char *dir, bool isLocal, int num);
void setMooringStiffness(double EA, double neutralStrain, int cabID, int num);

void getWindspeed(double posx, double posy, double posz, double *velocity);
void getWindspeedArray(double *posx, double *posy, double *posz, double *velx, double *vely, double *velz, int arraySize);
void getWaveVelAccElev(double posx, double posy, double posz, double *velocity, double *acceleration, double *elevation);
void getTowerBottomLoads_at_num(double *loads, int num);
void getTurbineOperation_at_num(double *vars, int num);
double getCustomData_at_num(const char *str, double pos, int num);
double getCustomSimulationTimeData(const char *str);