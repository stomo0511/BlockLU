CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		01_BlockLU.o

LIBS =

TARGET =	01_BlockLU

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
