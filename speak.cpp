#include "speak.h"
#include "kbhit.h"
#include <sstream>
#include <iostream>

static int _type = 0;
static std::string _language = "en";

void speak_init (int type, std::string language="en")
{
	_type = type;
	_language = language;
	switch (_type) {
	case 0:
		break;
	case 1:
		// TBD
		break;
	}	
}

void speak (std::string message)
{
	switch (_type) {
	case 0:
		std::cout << message << std::endl;
		break;
	case 1:
		//system(("espeak -v"+_language+" \""+message+"\" 2>/dev/null").c_str());
        // TBD
		break;
	}	
}

void speakYouAre (std::string name) 
{
	if (_language == "en") {
		speak("You are "+name);
	}
}

bool listen (std::string &message)
{
    static std::string str;
	switch (_type) {
	case 0:
		while (kbhit()) {
			int c = getch();
			if (c == 10 || c == 0);
			else if (c != 13) {
				char ch = static_cast<char>(c);
				str.push_back(ch);
				std::cout << ch << std::flush;
			}
			else {
				std::cout << std::endl;
				message = str;
				str = "";
				return true;
			}
		}
		break;
	case 1:
		// TBD
		break;
	}
    return false;
}

bool listenIam (std::string message, std::string &name)
{
	if (_language == "en") {
		std::istringstream str(message);
		std::string w1, w2, w3, w4;
		str >> w1;
		str >> w2;
		str >> w3;
		str >> w4;
		if ((w1 == "This" || w1 == "this") && w2 == "is" && w3 != "") {
			name = w3;
			return true;
		}
		if ((w1 == "I" || w1 == "i") && w2 == "am" && w3 != "") {
			name = w3;
			return true;
		}
		if ((w1 == "my" || w1 == "My" || w1 == "His" || w1 == "his" || w1 == "Her" || w1 == "her") && w2 == "name" && w3 == "is" && w4 != "") {
			name = w4;
			return true;
		}
	}
    return false;
}

bool listenWho (std::string message)
{
	if (_language == "en") {
		std::istringstream str(message);
		std::string w1;
		str >> w1;
		if (w1 == "Who" || w1 == "who") return true;
	}
    return false;
}
