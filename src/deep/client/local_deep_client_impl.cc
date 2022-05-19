// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chuan Cheng (chengchuancoder@gmail.com)
//

#include <memory>   // std::unique_ptr
#include <utility>  // std::move

#include "src/deep/client/deep_client_impl.h"
#include "src/deep/data_op/deep_op_factory.h"
#include "src/deep/data_op/deep_op_resource.h"
#include "src/deep/data_op/feature_lookuper_op/item_feature_lookuper.h"
#include "src/deep/data_op/instance_sampler_op/instance_sampler_op.h"
#include "src/deep/data_op/negative_sampler_op/shared_negative_sampler.h"
#include "src/deep/deep_config.h"
#include "src/deep/deep_data.h"
#include "src/sampler/sampler_builder.h"
#include "src/sampler/sampler_source.h"

namespace embedx {

class LocalDeepClientImpl : public DeepClientImpl {
 private:
  std::unique_ptr<deep_op::LocalDeepOpResource> resource_;
  deep_op::LocalDeepOpFactory* factory_ = nullptr;

 public:
  ~LocalDeepClientImpl() override = default;

 public:
  bool Init(const DeepConfig& config) override {
    resource_.reset(new deep_op::LocalDeepOpResource);

    resource_->set_deep_config(config);

    auto deep_data = DeepData::Create(config);
    if (!deep_data) {
      return false;
    }
    resource_->set_deep_data(std::move(deep_data));

    // The sampler needs to be created only if the freq file is not empty.
    if (!config.freq_file().empty()) {
      auto sampler_source = NewDeepSamplerSource(resource_->deep_data());
      if (!sampler_source) {
        return false;
      }
      resource_->set_sampler_source(std::move(sampler_source));

      auto negative_sampler_builder = NewSamplerBuilder(
          resource_->sampler_source(), SamplerBuilderEnum::NEGATIVE_SAMPLER,
          config.negative_sampler_type(), config.thread_num());
      if (!negative_sampler_builder) {
        return false;
      }
      resource_->set_negative_sampler_builder(
          std::move(negative_sampler_builder));
    }

    factory_ = deep_op::LocalDeepOpFactory::GetInstance();
    return factory_->Init(resource_.get());
  }

  bool SharedSampleNegative(
      int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
      std::vector<vec_int_t>* sampled_nodes_list) const override {
    auto* op = factory_->LookupOrCreate("SharedNegativeSampler");
    return dynamic_cast<deep_op::SharedNegativeSampler*>(op)->Run(
        count, nodes, excluded_nodes, sampled_nodes_list);
  }

  bool LookupItemFeature(const vec_int_t& items,
                         std::vector<vec_pair_t>* item_feats) const override {
    auto* op = factory_->LookupOrCreate("ItemFeatureLookuper");
    return dynamic_cast<deep_op::ItemFeatureLookuper*>(op)->Run(items,
                                                                item_feats);
  }

  bool SampleInstance(int count, vec_int_t* insts,
                      std::vector<vecl_t>* vec_labels_list) const override {
    auto* op = factory_->LookupOrCreate("InstanceSampler");
    return dynamic_cast<deep_op::InstanceSampler*>(op)->Run(count, insts,
                                                            vec_labels_list);
  }
};

std::unique_ptr<DeepClientImpl> NewLocalDeepClientImpl(
    const DeepConfig& config) {
  std::unique_ptr<DeepClientImpl> deep_client_impl;
  deep_client_impl.reset(new LocalDeepClientImpl());

  if (!deep_client_impl->Init(config)) {
    DXERROR("Failed to new local deep client impl.");
    deep_client_impl.reset();
  }

  return deep_client_impl;
}

}  // namespace embedx
