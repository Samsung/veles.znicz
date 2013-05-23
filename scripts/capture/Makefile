all: capture.exe capture64.exe

capture.exe: capture.c
	i686-w64-mingw32-gcc capture.c -std=c99 -pedantic -Wall -Werror -O3 -lgdi32 -o capture.exe

capture64.exe: capture.c
	x86_64-w64-mingw32-gcc capture.c -std=c99 -pedantic -Wall -Werror -O3 -lgdi32 -o capture64.exe

clean:
	rm -f capture.exe capture64.exe

