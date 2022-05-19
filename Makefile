# Tencent is pleased to support the open source community by making embedx
# available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Licensed under the BSD 3-Clause License and other third-party components,
# please refer to LICENSE for details.
#
# Author: Yafei Zhang (zhangyafeikimi@gmail.com)
#

SOURCE_ROOT  := $(shell pwd)
BUILD_ROOT   := $(shell pwd)
CC           ?= gcc
CXX          ?= g++
AR           ?= ar
CPPFLAGS     += \
-I$(SOURCE_ROOT)/deepx_core/include \
-I$(SOURCE_ROOT)/deepx_core/thirdparty \
-I$(SOURCE_ROOT)/include \
-I$(SOURCE_ROOT)
CFLAGS       += -pthread -std=c99 -g -Wall -Wextra -Werror -pedantic
CXXFLAGS     += -pthread -std=c++11 -g -Wall -Wextra -Werror -pedantic -Wno-out-of-line-declaration
LDFLAGS      += -pthread
MACHINE      := $(shell $(CXX) $(CPPFLAGS) $(CXXFLAGS) -dumpmachine)
BUILD_DIR    := build_$(MACHINE)
BUILD_DIR_ABS = $(BUILD_ROOT)/$(BUILD_DIR)

OS_DARWIN    := 0
OS_LINUX     := 0
OS_POSIX     := 0

ifeq ($(findstring darwin,$(MACHINE)),darwin)
OS_DARWIN    := 1
OS_POSIX     := 1
CPPFLAGS     += -DOS_DARWIN=1 -DOS_POSIX=1
endif

ifeq ($(findstring linux,$(MACHINE)),linux)
OS_LINUX     := 1
OS_POSIX     := 1
CPPFLAGS     += -DOS_LINUX=1 -DOS_POSIX=1
LDFLAGS      += -ldl
endif

DEBUG        ?= 0
ifneq ($(DEBUG),1)
CPPFLAGS     += -DNDEBUG
CFLAGS       += -O3
CXXFLAGS     += -O3
BUILD_DIR    := $(BUILD_DIR)_r
else
CFLAGS       += -O0
CXXFLAGS     += -O0 -Og
BUILD_DIR    := $(BUILD_DIR)_d
endif

ifneq ($(MAKECMDGOALS),lint)
CPPFLAGS     += -I$(BUILD_DIR_ABS) -DHAVE_COMPILE_FLAGS_H=1
endif

SOURCES      := $(shell find src -type f -name "*.cc" | sort)
TEST_SOURCES := $(shell find src -type f -name "*_test*.cc" | sort)
BIN_SOURCES  := $(shell find src -type f -name "*_main.cc" | sort)
FLAGS_SOURCES:= $(shell find src/tools/dist -type f -name "*.cc" | sort)
LIB_SOURCES  := $(filter-out $(TEST_SOURCES) $(BIN_SOURCES) $(FLAGS_SOURCES),$(SOURCES))
LINT_SOURCES := $(SOURCES:.cc=.lint)

TEST_OBJECTS := $(addprefix $(BUILD_DIR_ABS)/,$(TEST_SOURCES))
TEST_OBJECTS := $(TEST_OBJECTS:.cc=.o)
FLAGS_OBJECTS:= $(addprefix $(BUILD_DIR_ABS)/,$(FLAGS_SOURCES))
FLAGS_OBJECTS:= $(FLAGS_SOURCES:.cc=.o)
LIB_OBJECTS  := $(addprefix $(BUILD_DIR_ABS)/,$(LIB_SOURCES))
LIB_OBJECTS  := $(LIB_OBJECTS:.cc=.o)

ifeq ($(filter $(MAKECMDGOALS),clean lint),)
DEPENDS      := $(addprefix $(BUILD_DIR_ABS)/,$(SOURCES))
DEPENDS      := $(DEPENDS:.cc=.d)
else
DEPENDS      :=
endif

LIBRARIES    := $(BUILD_DIR_ABS)/libembedx.a
BINARIES     := \
	$(BUILD_DIR_ABS)/trainer \
	$(BUILD_DIR_ABS)/predictor \
	$(BUILD_DIR_ABS)/dist_trainer \
	$(BUILD_DIR_ABS)/unit_test \
	$(BUILD_DIR_ABS)/tools/graph/average_feature_main \
	$(BUILD_DIR_ABS)/tools/graph/graph_server_main \
	$(BUILD_DIR_ABS)/tools/graph/graph_client_main \
	$(BUILD_DIR_ABS)/tools/graph/close_server_main \
	$(BUILD_DIR_ABS)/tools/graph/random_walker_main \
	$(BUILD_DIR_ABS)/merge_model_shard \
	$(BUILD_DIR_ABS)/model_server_demo \

LIBS         := $(LIBRARIES)
TEST_LIBS    :=
ifeq ($(OS_DARWIN),1)
FORCE_LIBS   := -Wl,-all_load $(LIBRARIES)
else
FORCE_LIBS   := -Wl,--whole-archive $(LIBRARIES) -Wl,--no-whole-archive
endif

################################################################

all: $(LIBRARIES) $(BINARIES)
	@echo "******************************************"
	@echo "Build succsessfully at $(BUILD_DIR_ABS)"
	@echo "CC:          " $(CC)
	@echo "CXX:         " $(CXX)
	@echo "AR:          " $(AR)
	@echo "CPPFLAGS:    " $(CPPFLAGS)
	@echo "CFLAGS:      " $(CFLAGS)
	@echo "CXXFLAGS:    " $(CXXFLAGS)
	@echo "LDFLAGS:     " $(LDFLAGS)
	@echo "******************************************"
	@echo "DEBUG:       " $(DEBUG)
	@echo "******************************************"
.PHONY: all

clean: _clean clean_deepx_core
.PHONY: clean

_clean:
	@echo Cleaning $(BUILD_DIR_ABS)
	@rm -rf $(BUILD_DIR_ABS)
.PHONY: _clean

test: $(BUILD_DIR_ABS)/unit_test
	@cd $(BUILD_DIR_ABS) && ./unit_test
.PHONY: test

lint: $(LINT_SOURCES)
.PHONY: lint

build_dir:
	@echo $(BUILD_DIR)
.PHONY: build_dir

build_dir_abs:
	@echo $(BUILD_DIR_ABS)
.PHONY: build_dir_abs

################################################################

$(BUILD_DIR_ABS)/compile_flags.h:
	@echo Generating $@
	@mkdir -p $(@D)
	@bash $(SOURCE_ROOT)/deepx_core/script/compile_flags.sh \
		CC "$(CC)" \
		CXX "$(CXX)" \
		AR "$(AR)" \
		CPPFLAGS "$(CPPFLAGS)" \
		CFLAGS "$(CFLAGS)" \
		CXXFLAGS "$(CXXFLAGS)" \
		LDFLAGS "$(LDFLAGS)" \
		MACHINE "$(MACHINE)" \
		> $@

$(BUILD_DIR_ABS)/%.o: %.cc $(BUILD_DIR_ABS)/compile_flags.h
	@echo Compiling $<
	@mkdir -p $(@D)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR_ABS)/%.d: %.cc $(BUILD_DIR_ABS)/compile_flags.h
	@echo Scanning dependency $<
	@mkdir -p $(@D)
	@$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $< | sed -e 's,\(.*\)\.o[ :]*,$(@D)/\1.o $@: ,g' > $@

-include $(DEPENDS)

src/%.lint: src/%.cc
	@echo Linting"(clang-tidy)" $<
	@clang-tidy -quiet -header-filter='$(SOURCE_ROOT)/src/.*\.h' $< -- $(CPPFLAGS) $(CXXFLAGS)
.PHONY: src/%.lint

################################################################

DEEPX_CORE_LIBS   := $(SOURCE_ROOT)/deepx_core/$(BUILD_DIR)/libdeepx_core.a
DEEPX_GFLAGS_LIBS := $(SOURCE_ROOT)/deepx_core/$(BUILD_DIR)/libdeepx_gflags.a
DEEPX_GTEST_LIBS  := $(SOURCE_ROOT)/deepx_core/$(BUILD_DIR)/libdeepx_gtest.a
DEEPX_LZ4_LIBS    := $(SOURCE_ROOT)/deepx_core/$(BUILD_DIR)/libdeepx_lz4.a
DEEPX_Z_LIBS      := $(SOURCE_ROOT)/deepx_core/$(BUILD_DIR)/libdeepx_z.a

$(DEEPX_CORE_LIBS): build_deepx_core
$(DEEPX_GFLAGS_LIBS): build_deepx_core
$(DEEPX_GTEST_LIBS): build_deepx_core
$(DEEPX_LZ4_LIBS): build_deepx_core
$(DEEPX_Z_LIBS): build_deepx_core

build_deepx_core:
	@$(MAKE) -C deepx_core lib DEBUG=$(DEBUG)
.PHONY: build_deepx_core

clean_deepx_core:
	@$(MAKE) -s -C deepx_core clean DEBUG=$(DEBUG)
.PHONY: clean_deepx_core

LIBS         += $(DEEPX_CORE_LIBS) $(DEEPX_GFLAGS_LIBS) $(DEEPX_LZ4_LIBS) $(DEEPX_Z_LIBS)
TEST_LIBS    += $(DEEPX_GTEST_LIBS)
ifeq ($(OS_DARWIN),1)
FORCE_LIBS   += -Wl,-all_load $(DEEPX_CORE_LIBS)
else
FORCE_LIBS   += -Wl,--whole-archive $(DEEPX_CORE_LIBS) -Wl,--no-whole-archive
endif

################################################################

$(BUILD_DIR_ABS)/libembedx.a: $(LIB_OBJECTS)
	@echo Archiving $@
	@mkdir -p $(@D)
	@$(AR) rcs $@ $^

$(BUILD_DIR_ABS)/dist_trainer: \
	$(BUILD_DIR_ABS)/src/tools/dist/dist_trainer_main.o \
	$(BUILD_DIR_ABS)/src/tools/dist/dist_flags.o \
	$(BUILD_DIR_ABS)/src/tools/dist/dist_server.o \
	$(BUILD_DIR_ABS)/src/tools/dist/dist_worker.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/trainer: \
	$(BUILD_DIR_ABS)/src/tools/trainer_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/predictor: \
	$(BUILD_DIR_ABS)/src/tools/predictor_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/tools/graph/graph_server_main: \
	$(BUILD_DIR_ABS)/src/tools/graph/dist_graph_server_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/tools/graph/graph_client_main: \
	$(BUILD_DIR_ABS)/src/tools/graph/dist_graph_client_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/tools/graph/close_server_main: \
	$(BUILD_DIR_ABS)/src/tools/graph/close_server_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/tools/graph/random_walker_main: \
	$(BUILD_DIR_ABS)/src/tools/graph/random_walker_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/tools/graph/average_feature_main: \
	$(BUILD_DIR_ABS)/src/tools/graph/average_feature_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/unit_test: \
	$(TEST_OBJECTS) \
	$(LIBS) \
	$(TEST_LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	rm -rf $(@D)/testdata
	cp -r src/testdata $(@D)/testdata
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/merge_model_shard: \
	$(BUILD_DIR_ABS)/deepx_core/src/tools/merge_model_shard_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)

$(BUILD_DIR_ABS)/model_server_demo: \
	$(BUILD_DIR_ABS)/src/tools/model_server_demo_main.o \
	$(LIBS)
	@echo Linking $@
	@mkdir -p $(@D)
	@$(CXX) -o $@ $(FORCE_LIBS) $^ $(LDFLAGS)
