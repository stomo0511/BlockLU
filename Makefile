CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		BlockLU.o

LIBS =

TARGET =	BlockLU

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
