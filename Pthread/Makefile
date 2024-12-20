PTHREAD = pthread
# 設定 C++ 編譯器
CXX := g++

# 設定編譯選項
CXXFLAGS := -O3 -std=c++17 -Wall -pthread

# 設定包含目錄
INCLUDE_DIR := include

# 設定源碼檔案和目標檔案
SRCS := jpeg_${PTHREAD}.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := JPEGCompress_main

# 預設目標
all: $(TARGET)

# 編譯目標
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# 編譯源碼檔案
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# 清理編譯產物
clean:
	$(RM) $(OBJS) $(TARGET) *~

# 幫助目標
help:
	@echo "Makefile targets:"
	@echo "  all      - 編譯程式"
	@echo "  clean    - 清理編譯產物"
	@echo "  help     - 顯示此幫助信息"