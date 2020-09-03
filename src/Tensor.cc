#include "Tensor.h"

#include <exception>
#include "constants.h"
#include "utils.h"

namespace torchjs
{

  using namespace constants;

  namespace
  {
    template <typename T>
    Napi::Value tensorToArray(Napi::Env env, const torch::Tensor &tensor)
    {
      try
      {
        Napi::EscapableHandleScope scope(env);
        assert(tensor.is_contiguous());
        auto typed_array = Napi::TypedArrayOf<T>::New(env, tensor.numel());
        memcpy(typed_array.Data(), tensor.data<T>(), sizeof(T) * tensor.numel());
        auto shape_array = tensorShapeToArray(env, tensor);
        auto obj = Napi::Object::New(env);
        obj.Set(kData, typed_array);
        obj.Set(kShape, shape_array);
        return scope.Escape(obj);
      }
      catch (const std::exception &e)
      {
        throw Napi::Error::New(env, e.what());
      }
    }

    template <typename T>
    Napi::Value arrayToTensor(Napi::Env env, const Napi::TypedArray &data,
                              const ShapeArrayType &shape_array)
    {
      try
      {
        Napi::EscapableHandleScope scope(env);
        auto *data_ptr = data.As<Napi::TypedArrayOf<T>>().Data();
        auto shape = shapeArrayToVector(shape_array);
        torch::TensorOptions options(scalarType<T>());
        auto torch_tensor = torch::empty(shape, options);
        memcpy(torch_tensor.data<T>(), data_ptr, sizeof(T) * torch_tensor.numel());
        return scope.Escape(Tensor::FromTensor(env, torch_tensor));
      }
      catch (const std::exception &e)
      {
        throw Napi::Error::New(env, e.what());
      }
    }
  } // namespace

  Napi::Object Tensor::Init(Napi::Env env, Napi::Object exports)
  {
    Napi::Function func = DefineClass(env, "Tensor",
                                      {
                                          InstanceMethod("toString", &Tensor::toString),
                                          InstanceMethod("toObject", &Tensor::toObject),
                                          StaticMethod("fromObject", &Tensor::fromObject),
                                          InstanceMethod("cpu", &Tensor::cpu),
                                          InstanceMethod("cuda", &Tensor::cuda),
                                          InstanceMethod("free", &Tensor::free),
                                      });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("Tensor", func);
    return exports;
  }

  Napi::Object Tensor::FromTensor(Napi::Env env, const torch::Tensor &tensor)
  {
    try
    {
      Napi::EscapableHandleScope scope(env);
      auto obj = constructor.New({});
      Napi::ObjectWrap<Tensor>::Unwrap(obj)->tensor_ = tensor;
      return scope.Escape(obj).ToObject();
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(env, e.what());
    }
  }

  Tensor::Tensor(const Napi::CallbackInfo &info) : ObjectWrap<Tensor>(info) {}

  Napi::FunctionReference Tensor::constructor;

  torch::Tensor Tensor::tensor()
  {
    return tensor_;
  }

  bool Tensor::IsInstance(Napi::Object &obj)
  {
    return obj.InstanceOf(constructor.Value());
  }

  Napi::Value Tensor::toString(const Napi::CallbackInfo &info)
  {
    try
    {
      return Napi::String::New(info.Env(), tensor_.toString());
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  Napi::Value Tensor::toObject(const Napi::CallbackInfo &info)
  {
    try
    {
      auto env = info.Env();
      auto st = tensor_.scalar_type();
      switch (st)
      {
      case torch::ScalarType::Float:
        return tensorToArray<float>(env, tensor_);
      case torch::ScalarType::Double:
        return tensorToArray<double>(env, tensor_);
      case torch::ScalarType::Int:
        return tensorToArray<int32_t>(env, tensor_);
      default:
        throw Napi::TypeError::New(env, "Unsupported type");
      }
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  Napi::Value Tensor::fromObject(const Napi::CallbackInfo &info)
  {
    try
    {
      auto env = info.Env();
      Napi::HandleScope scope(env);
      auto obj = info[0].As<Napi::Object>();
      auto data = obj.Get(kData).As<Napi::TypedArray>();
      auto shape = obj.Get(kShape).As<ShapeArrayType>();
      auto data_type = data.TypedArrayType();
      switch (data_type)
      {
      case napi_float32_array:
        return arrayToTensor<float>(env, data, shape);
      case napi_float64_array:
        return arrayToTensor<double>(env, data, shape);
      case napi_int32_array:
        return arrayToTensor<int32_t>(env, data, shape);
      default:
        throw Napi::TypeError::New(env, "Unsupported type");
      }
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  Napi::Value Tensor::cpu(const Napi::CallbackInfo &info)
  {
    try
    {
      return FromTensor(info.Env(), tensor_.cpu());
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  Napi::Value Tensor::cuda(const Napi::CallbackInfo &info)
  {
    try
    {
      if (torch::cuda::is_available())
      {
        return FromTensor(info.Env(), tensor_.cuda());
      }
      else
      {
        throw Napi::Error::New(info.Env(), "CUDA is not aviliable");
      }
      return info.This();
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  void Tensor::Finalize(Napi::Env env)
  {
    tensor_.getIntrusivePtr()->release_resources();
  }

  void Tensor::free(const Napi::CallbackInfo &info)
  {
    tensor_.getIntrusivePtr()->release_resources();
  }

} // namespace torchjs