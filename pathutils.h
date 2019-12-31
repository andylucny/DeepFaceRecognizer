#ifndef PATHUTILS_H
#define PATHUTILS_H

#include <string>
#include <stdio.h>
#if _WIN32
#include <direct.h>  
#include <stdlib.h>  
#define mkdir(path,mode) _mkdir(path)
#else
#include <sys/types.h> 
#include <sys/stat.h>
#endif

inline void createpath (std::string path)
{
    mkdir (path.c_str(),0777);
}

inline bool exists (const std::string& name) {
	struct stat buffer;   
	return (stat (name.c_str(), &buffer) == 0); 
}

inline void deletepath (std::string path)
{
	remove(path.c_str());
}

#endif //PATHUTILS_H

