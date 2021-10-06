/////////////////////////////////////////////////////////////////////////////
// Copyright (c) Electronic Arts Inc. All rights reserved.
/////////////////////////////////////////////////////////////////////////////

#ifndef EASTL_INTERNAL_TRACED_H
#define EASTL_INTERNAL_TRACED_H

#include <EASTL/internal/config.h>

#include <EASTL/utility.h> /* eastl::swap */

#include <cstddef> /* size_t */
#include <cstring> /* memcpy/move/set */
#include <type_traits> /* remove_pointer_t */
#include <utility> /* move */

////////////////////////////////////////////////////////////////////////////////////////////
// Support for tracked writes.
////////////////////////////////////////////////////////////////////////////////////////////

namespace eastl
{
	/*
	 * Helper for empty tracker classes. A Tracker is typically an empty class,
	 * and has a protected destructor to avoid -Wefc++ complaints when inhereted.
	 * Destructible wraps the Tracker to allow a stand-alone Tracker object.
	 */
	template <typename Tracker>
		struct Destructible
			: public Tracker
		{};

	template <typename Tracker>
		class Modifier
			: private Tracker
		{
			const void *const _p;
			const std::size_t _s;
			const char _id;

		public:
			Modifier(const Tracker &t_, const void *p_, std::size_t s_, char id_ = '?')
				: Tracker(t_)
				, _p(p_)
				, _s(s_)
				, _id(id_)
			{
				this->track_pre(_p, _s, _id);
			}
			~Modifier()
			{
				this->track_post(_p, _s, _id);
			}
		};

	template <typename Tracker>
		Modifier<Tracker> make_modifier_range(const Tracker &r_, const void *p_, std::size_t s_, char id_ = '?')
		{
			return Modifier<Tracker>(r_, p_, s_, id_);
		}

	template <typename Tracker, typename Object>
		Modifier<Tracker> make_modifier_n(const Tracker &r_, const Object &o_, std::size_t n, char id_ = '?')
		{
			return make_modifier_range(r_, &o_, n * (sizeof o_), id_);
		}

	template <typename Tracker, typename Object>
		Modifier<Tracker> make_modifier(const Tracker &r_, const Object &o_, char id_ = '?')
		{
			return make_modifier_n(r_, o_, 1, id_);
		}

	template <typename Tracker>
		class MemWrapper
			: public Tracker
		{
		public:
			MemWrapper(const Tracker &r_)
				: Tracker(r_)
			{}
			auto memcpy(void *dst, const void *src, std::size_t ct) const
			{
				/* TODO: should write an object to coordinate calls to track_pre and track_post. */
				auto m = make_modifier_range(*this, dst, ct);
				return ::memcpy(dst, src, ct);
			}

			auto memmove(void *dst, const void *src, std::size_t ct) const
			{
				auto m = make_modifier_range(*this, dst, ct);
				return ::memmove(dst, src, ct);
			}

			auto memset(void *dst, int c, std::size_t ct) const
			{
				auto m = make_modifier_range(*this, dst, ct);
				return ::memset(dst, c, ct);
			}
		};

	template <typename Tracker>
		MemWrapper<Tracker> mem_wrap(const Tracker &r_)
		{
			return MemWrapper<Tracker>(r_);
		}

	class tracked_temp{};

	/*
	 * Templates for values whose access to memory must be tracked.
	 */

	/*
	 * A "weak_tracked" does not include the tracker. It is used
	 * only when test cases notice that using a regular tracker
	 * would inhibit empty base class optimization.
	 */

	template <typename Type, typename Tracker, char ID='?'>
		class weak_tracked
		{
		public:
			using modifier_type = Modifier<Tracker>;
		private:
			/* Note: consider making type privately inherited if it is a class/struct,
			 * to accomodate empty classes
			 */
			Type _t;

			void track_0(const Tracker &r_) noexcept
			{
				r_.track_pre(&_t, sizeof _t, ID);
			}
			void track_1(const Tracker &r_) noexcept
			{
				r_.track_post(&_t, sizeof _t, ID);
			}

			/* internal constructor */
			weak_tracked(const Type &t_, tracked_temp) noexcept
				: _t(t_)
			{
			}

		public:
			/* "Rule of five" declarations */
			/* If this fails because there is no suitable Destructible<Tracker> constructor,
			 * the tracker presumably has internal state and cannot be used with the
			 * specifed type, e.g. deque, string, string_view.
			 */
			weak_tracked(const Tracker &r_, const Type &t_) noexcept
				: _t(t_)
			{
				/* Constructors are an exception to the rule that all tracking
				 * must include a pre and post call. Creation of data out of
				 * raw bytes should not care about the previous value of the
				 * raw bytes.
				 */
				track_1(r_);
			}

			weak_tracked(const Tracker &, const weak_tracked<Type, Tracker, ID> &other_) noexcept
				: _t(other_._t)
			{
				track_1();
			}

			weak_tracked(const Tracker &r_, weak_tracked<Type, Tracker, ID> &&other_) noexcept
				: _t((other_.track_0(r_), std::move(other_._t)))
			{
				track_1(r_);
				/* the source may have been altered too */
				other_.track_1(r_);
			}
			weak_tracked &operator=(const Type &t_) noexcept = delete;

			modifier_type make_modifier(const Tracker &r_)
			{
				return modifier_type(r_, &_t, sizeof _t, ID);
			}

			weak_tracked &assign(const Tracker &r_, const Type &t_) noexcept
			{
				auto m = make_modifier(r_);
				_t = t_;
				return *this;
			}

			~weak_tracked() = default;

			/* zero-argument constructor */
			weak_tracked() = default;
			/* Read access to the value */
			operator Type() const
			{
				return _t;
			}
			/* explicit read access */
			const Type &value() const
			{
				return _t;
			}
			/* explicit write access */
			Type &value(const modifier_type &)
			{
				return _t;
			}

			/* pointers */
			std::remove_pointer_t<const Type> &operator*() const
			{
				return *_t;
			}

			const Type &operator->() const
			{
				return _t;
			}

			template <typename U>
				weak_tracked<U *, Tracker, ID> ptr_cast() const
				{
					return weak_tracked<U *, Tracker, ID>(static_cast<U *>(_t), tracked_temp{}, *this);
				}

			template <typename U, typename V, char I>
				void swap(const Tracker & r_, weak_tracked<U, V, I> &b) noexcept
				{
					eastl::swap(_t, b._t);
					track(r_);
					b.track(r_);
				}

			template <typename U, typename V, char I>
				friend class eastl::weak_tracked;

			void swap(weak_tracked<Type, Tracker, ID> &a, weak_tracked<Type, Tracker, ID> &b) noexcept;
		};

	template <typename Type, typename Tracker, char ID>
		bool operator==(const weak_tracked<Type, Tracker, ID> &a, const weak_tracked<Type, Tracker, ID> &b)
		{
			return Type(a) == Type(b);
		}

	template <typename T, typename Tracker, char ID>
		struct iterator_traits<weak_tracked<T*, Tracker, ID>>
		{
			typedef EASTL_ITC_NS::random_access_iterator_tag iterator_category;
			typedef T                                        value_type;
			typedef ptrdiff_t                                difference_type;
			typedef T*                                       pointer;
			typedef T&                                       reference;
		};

	template <typename T, typename Tracker, char ID>
		struct iterator_traits<weak_tracked<const T*, Tracker, ID>>
		{
			typedef EASTL_ITC_NS::random_access_iterator_tag iterator_category;
			typedef T                                        value_type;
			typedef ptrdiff_t                                difference_type;
			typedef const T*                                 pointer;
			typedef const T&                                 reference;
		};

	template <typename Type, typename Tracker, char ID>
        inline void swap(const Tracker &r, weak_tracked<Type, Tracker, ID> &a, weak_tracked<Type, Tracker, ID> &b) noexcept
		{
			a.swap(r, b);
		}

	/* strong tracked */
	template <typename Type, typename Tracker, char ID='?', bool IsClass = std::is_class<Type>::value>
		struct value_tracked;

	template<typename Type, typename Tracker, char ID>
		struct value_tracked<Type, Tracker, ID, false>
			: public Tracker
		{
			using modifier_type = Modifier<Tracker>;
			using tracker_type = Tracker;
		private:
			/* Note: consider making type privately inherited if it is a class/struct,
			 * to accomodate empty classes
			 */
			Type _t;

			void track_0() noexcept
			{
				Tracker::track_pre(&_t, sizeof _t, ID);
			}
			void track_1() noexcept
			{
				Tracker::track_post(&_t, sizeof _t, ID);
			}

			/* internal constructor */
			value_tracked(const Type &t_, tracked_temp, const Tracker &r_) noexcept
				: Tracker(r_)
				, _t(t_)
			{
			}

		public:
			/* "Rule of five" declarations */
			/* If this fails because there is no suitable Destructible<Tracker> constructor,
			 * the tracker presumably has internal state and cannot be used with the
			 * specifed type, e.g. deque, string, string_view.
			 */
			value_tracked(const Type &t_, const Tracker &r_ = Destructible<Tracker>()) noexcept
				: Tracker(r_)
				, _t(t_)
			{
				/* Constructors are an exception to the rule that all tracking
				 * must include a pre and post call. Creation of data out of
				 * raw bytes should not care about the previous value of the
				 * raw bytes.
				 */
				track_1();
			}

			value_tracked(const value_tracked<Type, Tracker, ID> &other_) noexcept
				: Tracker(other_)
				, _t(other_._t)
			{
				track_1();
			}

			value_tracked(value_tracked<Type, Tracker, ID> &&other_) noexcept
				: Tracker(other_)
				, _t((other_.track_0(), std::move(other_._t)))
			{
				track_1();
				/* the source may have been altered too */
				other_.track_1();
			}

			modifier_type make_modifier()
			{
				return modifier_type(*this, &_t, sizeof _t, ID);
			}

			value_tracked &operator=(const Type &t_) noexcept
			{
				auto m = make_modifier();
				_t = t_;
				return *this;
			}

			/* ERROR: why not subsumed by the looser, templated version, which immediately follows? */
			value_tracked &operator=(const value_tracked<Type, Tracker, ID> &other_) noexcept
			{
				auto m = make_modifier();
				_t = other_._t;
				return *this;
			}

			template <typename U, char I>
				value_tracked &operator=(const value_tracked<Type, U, I> &other_) noexcept
				{
					auto m = make_modifier();
					_t = other_._t;
					return *this;
				}

			~value_tracked() = default;

			/* zero-argument constructor */
			value_tracked() = default;
			/* Read access to the value */
			operator Type() const
			{
				return _t;
			}
			/* explicit read access */
			const Type &value() const
			{
				return _t;
			}
			/* explicit write access */
			Type &value(const modifier_type &)
			{
				return _t;
			}
			/* scalars */
			value_tracked<Type, Tracker, ID> &operator++()
			{
				auto m = make_modifier();
				++_t;
				return *this;
			}
			template <typename U>
				value_tracked<Type, Tracker, ID> &operator+=(const U &u)
				{
					auto m = make_modifier();
					_t += u;
					return *this;
				}
			template <typename U>
				value_tracked<Type, Tracker, ID> &operator-=(const U &u)
				{
					auto m = make_modifier();
					_t -= u;
					return *this;
				}
			Type operator++(int)
			{
				auto t = _t;
				auto m = make_modifier();
				++_t;
				return t;
			}
			value_tracked<Type, Tracker, ID> &operator--()
			{
				auto m = make_modifier();
				--_t;
				return *this;
			}
			Type operator--(int)
			{
				auto t = _t;
				auto m = make_modifier();
				--_t;
				return t;
			}

			/* pointers */
			std::remove_pointer_t<const Type> &operator*() const
			{
				return *_t;
			}

			std::remove_pointer_t<Type> &deref(modifier_type &)
			{
				/* presume that the non-const version is the target of a store */
				return *_t;
			}

			const Type &operator->() const
			{
				return _t;
			}

			template <typename U>
				value_tracked<U *, Tracker, ID> ptr_cast() const
				{
					return value_tracked<U *, Tracker, ID>(static_cast<U *>(_t), tracked_temp{}, *this);
				}

			template <typename U, typename V, char I>
				void swap(value_tracked<U, V, I> &b) noexcept
				{
					auto m = make_modifier();
					auto mb = b.make_modifier();
					eastl::swap(_t, b._t);
				}

			template <typename U, typename V, char I>
				friend class eastl::value_tracked;

			void swap(value_tracked<Type, Tracker, ID> &a, value_tracked<Type, Tracker, ID> &b) noexcept;
		};

	template<typename Type, typename Tracker, char ID>
		struct value_tracked<Type, Tracker, ID, true>
			: public Tracker
			, public Type
		{
			using modifier_type = Modifier<Tracker>;
			using tracker_type = Tracker;
		private:
#if 0
			/* Note: consider making type privately inherited if it is a class/struct,
			 * to accomodate empty classes
			 */
			Type _t;
#endif
			void track_0() noexcept
			{
				auto &t = static_cast<const Type &>(*this);
				Tracker::track_pre(&t, sizeof t, ID);
			}
			void track_1() noexcept
			{
				auto &t = static_cast<const Type &>(*this);
				Tracker::track_post(&t, sizeof t, ID);
			}

			/* internal constructor */
			value_tracked(const Type &t_, tracked_temp, const Tracker &r_) noexcept
				: Tracker(r_)
				, Type(t_)
			{
			}

		public:
			/* "Rule of five" declarations */
			/* If this fails because there is no suitable Destructible<Tracker> constructor,
			 * the tracker presumably has internal state and cannot be used with the
			 * specifed type, e.g. deque, string, string_view.
			 */
			value_tracked(const Type &t_, const Tracker &r_ = Destructible<Tracker>()) noexcept
				: Tracker(r_)
				, Type(t_)
			{
				/* Constructors are an exception to the rule that all tracking
				 * must include a pre and post call. Creation of data out of
				 * raw bytes should not care about the previous value of the
				 * raw bytes.
				 */
				track_1();
			}

			value_tracked(const value_tracked<Type, Tracker, ID> &other_) noexcept
				: Tracker(other_)
				, Type(other_)
			{
				track_1();
			}

			value_tracked(value_tracked<Type, Tracker, ID> &&other_) noexcept
				: Tracker(other_)
				, Type((other_.track_0(), std::move(other_.t())))
			{
				track_1();
				/* the source may have been altered too */
				other_.track_1();
			}

			modifier_type make_modifier()
			{
				return make_modifier(*this, value(), ID);
			}

			value_tracked &operator=(const Type &t_) noexcept
			{
				auto m = make_modifier();
				value(make_modifier()) = t_;
				return *this;
			}

			/* ERROR: why not subsumed by the looser, templated version, which immediately follows? */
			value_tracked &operator=(const value_tracked<Type, Tracker, ID> &other_) noexcept
			{
				auto m = make_modifier();
				value(make_modifier()) = other_.t();
				return *this;
			}

			template <typename U, char I>
				value_tracked &operator=(const value_tracked<Type, U, I> &other_) noexcept
				{
					auto m = make_modifier();
					value(make_modifier()) = other_.t();
					return *this;
				}

			~value_tracked() = default;

			/* zero-argument constructor */
			value_tracked() = default;
			/* Read access to the value */
			operator Type() const
			{
				return *this;
			}
			/* explicit read access */
			const Type &value() const
			{
				return *this;
			}
			/* explicit write access */
			Type &value(const modifier_type &)
			{
				return *this;
			}
			/* scalars */
			value_tracked<Type, Tracker, ID> &operator++()
			{
				auto m = make_modifier();
				++value(m);
				return *this;
			}
			template <typename U>
				value_tracked<Type, Tracker, ID> &operator+=(const U &u)
				{
					auto m = make_modifier();
					value(m) += u;
					return *this;
				}
			template <typename U>
				value_tracked<Type, Tracker, ID> &operator-=(const U &u)
				{
					auto m = make_modifier();
					value(m) -= u;
					return *this;
				}
			Type operator++(int)
			{
				auto t = value();
				auto m = make_modifier();
				++value(m);
				return t;
			}
			value_tracked<Type, Tracker, ID> &operator--()
			{
				auto m = make_modifier();
				--value(m);
				return *this;
			}
			Type operator--(int)
			{
				auto t = value();
				auto m = make_modifier();
				--value(m);
				return t;
			}

			/* pointers */
			std::remove_pointer_t<const Type> &operator*() const
			{
				return *value();
			}

			std::remove_pointer_t<Type> &deref(modifier_type &m)
			{
				/* presume that the non-const version is the target of a store */
				return *value(m);
			}

			const Type &operator->() const
			{
				return value();
			}

			template <typename U>
				value_tracked<U *, Tracker, ID> ptr_cast() const
				{
					return value_tracked<U *, Tracker, ID>(static_cast<U *>(value()), tracked_temp{}, *this);
				}

			template <typename U, typename V, char I>
				void swap(value_tracked<U, V, I> &b) noexcept
				{
					auto m = make_modifier();
					auto mb = b.make_modifier();
					eastl::swap(value(m), b.value(mb));
				}

			template <typename U, typename V, char I>
				friend class eastl::value_tracked;

			void swap(value_tracked<Type, Tracker, ID> &a, value_tracked<Type, Tracker, ID> &b) noexcept;
		};

	template <typename Type, typename Tracker, char ID>
		bool operator==(const value_tracked<Type, Tracker, ID> &a, const value_tracked<Type, Tracker, ID> &b)
		{
			return Type(a) == Type(b);
		}

	template <typename T, typename Tracker, char ID>
		struct iterator_traits<value_tracked<T*, Tracker, ID>>
		{
			typedef EASTL_ITC_NS::random_access_iterator_tag iterator_category;
			typedef T                                        value_type;
			typedef ptrdiff_t                                difference_type;
			typedef T*                                       pointer;
			typedef T&                                       reference;
		};

	template <typename T, typename Tracker, char ID>
		struct iterator_traits<value_tracked<const T*, Tracker, ID>>
		{
			typedef EASTL_ITC_NS::random_access_iterator_tag iterator_category;
			typedef T                                        value_type;
			typedef ptrdiff_t                                difference_type;
			typedef const T*                                 pointer;
			typedef const T&                                 reference;
		};

	template <typename Type, typename Tracker, char ID>
		inline void swap(value_tracked<Type, Tracker, ID> &a, value_tracked<Type, Tracker, ID> &b) noexcept
		{
			a.swap(b);
		}

} // namespace eastl

#endif // EASTL_INTERNAL_TRACED_H

