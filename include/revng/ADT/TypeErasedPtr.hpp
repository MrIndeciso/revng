#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

/// Unique smart pointer with a type-erased deleter. Points to an arbitrary
/// T object which is deleted on destruction by the type-erased deleter.
///
/// The free helper function makeTypeErased can be used to allocate an object
/// on the heap and managed by a TypeErasedPtr.
///
/// The free helper function borrowTypeErased can be used to create a borrowing
/// TypeErasedPtr with a no-op deleter.
///
/// The aliasing constructor can be used to create a pointer to a subobject of
/// a larger object, or even to an object indirectly owned by an object managed
/// using a TypeErasedPtr. Pass the TypeErasedPtr along with a pointer to an
/// object whose lifetime is tied to the lifetime of the TypeErasedPtr.
///
/// // P1 points to a pair containing subobjects of type T and U.
/// TypeErasedPtr<std::pair<T, U>> P1 = makeTypeErased<std::pair<T, U>>(...);
///
/// // P2 hides the pair and points to its first subobject of type T.
/// TypeErasedPtr<T> P2 = TypeErasedPtr<T>(std::move(P1), &P1->first);
template<typename T>
class TypeErasedPtr : private std::shared_ptr<T> {
  template<typename>
  friend class TypeErasedPtr;

public:
  TypeErasedPtr() = default;
  TypeErasedPtr(decltype(nullptr)) noexcept {}

  template<typename U, typename DeleterT>
    requires std::convertible_to<U *, T *> and std::invocable<DeleterT &, T *>
  explicit TypeErasedPtr(U *Pointer, DeleterT Deleter) :
    std::shared_ptr<T>(Pointer, std::move(Deleter)) {}

  template<typename U>
  explicit TypeErasedPtr(TypeErasedPtr<U> &&Owner, T *Pointer) noexcept :
    std::shared_ptr<T>(static_cast<std::shared_ptr<U> &&>(Owner), Pointer) {}

  TypeErasedPtr(TypeErasedPtr &&) noexcept = default;
  TypeErasedPtr &operator=(TypeErasedPtr &&) noexcept = default;

  TypeErasedPtr(const TypeErasedPtr &) = delete;
  TypeErasedPtr &operator=(const TypeErasedPtr &) = delete;

  using std::shared_ptr<T>::reset;
  using std::shared_ptr<T>::swap;

  using std::shared_ptr<T>::get;
  using std::shared_ptr<T>::operator*;
  using std::shared_ptr<T>::operator->;
  using std::shared_ptr<T>::operator bool;

  template<typename... ArgsT>
  static TypeErasedPtr make(ArgsT &&...Args) {
    return TypeErasedPtr(std::make_shared<T>(std::forward<ArgsT>(Args)...));
  }

private:
  explicit TypeErasedPtr(std::shared_ptr<T> Pointer) :
    std::shared_ptr<T>(std::move(Pointer)) {}
};

template<typename T, typename... ArgsT>
[[nodiscard]] TypeErasedPtr<T> makeTypeErased(ArgsT &&...Args) {
  return TypeErasedPtr<T>::make(std::forward<ArgsT>(Args)...);
}

template<typename T>
[[nodiscard]] TypeErasedPtr<T> borrowTypeErased(T *Pointer) {
  return TypeErasedPtr<T>(Pointer, [](T *Pointer) {});
}
