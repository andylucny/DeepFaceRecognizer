#ifndef SPEAK_H
#define SPEAK_H

#include <string>

void speak_init (int type, std::string language);
void speak (std::string message);
void speakYouAre (std::string name);
bool listen(std::string &message);
bool listenIam (std::string message, std::string &name);
bool listenWho (std::string message);

#endif //SPEAK_H