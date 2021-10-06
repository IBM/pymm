///////////////////////////////////////////////////////////////////////////////
// Copyright (c) Electronic Arts Inc. All rights reserved.
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// The tree insert and erase functions below are based on the original 
// HP STL tree functions. Use of these functions was been approved by
// EA legal on November 4, 2005 and the approval documentation is available
// from the EASTL maintainer or from the EA legal deparatment on request.
// 
// Copyright (c) 1994
// Hewlett-Packard Company
// 
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation. Hewlett-Packard Company makes no
// representations about the suitability of this software for any
// purpose. It is provided "as is" without express or implied warranty.
///////////////////////////////////////////////////////////////////////////////




#include <stddef.h>



namespace eastl
{
	// Forward declarations
	template <typename Allocator>
	rbtree_node_base<Allocator>* RBTreeRotateLeft(rbtree_node_base<Allocator>* pNode, rbtree_node_base<Allocator>* pNodeRoot);
	template <typename Allocator>
	rbtree_node_base<Allocator>* RBTreeRotateRight(rbtree_node_base<Allocator>* pNode, rbtree_node_base<Allocator>* pNodeRoot);



	/// RBTreeIncrement
	/// Returns the next item in a sorted red-black tree.
	///
	template <typename Allocator>
	rbtree_node_base<Allocator>* RBTreeIncrement(const rbtree_node_base<Allocator>* pNode)
	{
		if(pNode->mpNodeRight) 
		{
			pNode = pNode->mpNodeRight;

			while(pNode->mpNodeLeft)
				pNode = pNode->mpNodeLeft;
		}
		else 
		{
			rbtree_node_base<Allocator>* pNodeTemp = pNode->mpNodeParent;

			while(pNode == pNodeTemp->mpNodeRight) 
			{
				pNode = pNodeTemp;
				pNodeTemp = pNodeTemp->mpNodeParent;
			}

			if(pNode->mpNodeRight != pNodeTemp)
				pNode = pNodeTemp;
		}

		return const_cast<rbtree_node_base<Allocator>*>(pNode);
	}



	/// RBTreeIncrement
	/// Returns the previous item in a sorted red-black tree.
	///
	template <typename Allocator>
	EASTL_API rbtree_node_base<Allocator>* RBTreeDecrement(const rbtree_node_base<Allocator>* pNode)
	{
		if((pNode->mpNodeParent->mpNodeParent == pNode) && (pNode->mColor == RBTreeColor::kRed))
			return pNode->mpNodeRight;
		else if(pNode->mpNodeLeft)
		{
			rbtree_node_base<Allocator>* pNodeTemp = pNode->mpNodeLeft;

			while(pNodeTemp->mpNodeRight)
				pNodeTemp = pNodeTemp->mpNodeRight;

			return pNodeTemp;
		}

		rbtree_node_base<Allocator>* pNodeTemp = pNode->mpNodeParent;

		while(pNode == pNodeTemp->mpNodeLeft) 
		{
			pNode     = pNodeTemp;
			pNodeTemp = pNodeTemp->mpNodeParent;
		}

		return const_cast<rbtree_node_base<Allocator>*>(pNodeTemp);
	}



	/// RBTreeGetBlackCount
	/// Counts the number of black nodes in an red-black tree, from pNode down to the given bottom node.  
	/// We don't count red nodes because red-black trees don't really care about
	/// red node counts; it is black node counts that are significant in the 
	/// maintenance of a balanced tree.
	///
	template <typename Allocator>
	EASTL_API size_t RBTreeGetBlackCount(const rbtree_node_base<Allocator>* pNodeTop, const rbtree_node_base<Allocator>* pNodeBottom)
	{
		size_t nCount = 0;

		for(; pNodeBottom; pNodeBottom = pNodeBottom->mpNodeParent)
		{
			if(pNodeBottom->mColor == RBTreeColor::kBlack)
				++nCount;

			if(pNodeBottom == pNodeTop) 
				break;
		}

		return nCount;
	}


	/// RBTreeRotateLeft
	/// Does a left rotation about the given node. 
	/// If you want to understand tree rotation, any book on algorithms will
	/// discussion the topic in good detail.
	template <typename Allocator>
	rbtree_node_base<Allocator>* RBTreeRotateLeft(rbtree_node_base<Allocator>* pNode, rbtree_node_base<Allocator>* pNodeRoot)
	{
		rbtree_node_base<Allocator>* const pNodeTemp = pNode->mpNodeRight;

		pNode->mpNodeRight = pNodeTemp->mpNodeLeft;

		if(pNodeTemp->mpNodeLeft)
			pNodeTemp->mpNodeLeft->mpNodeParent = pNode;
		pNodeTemp->mpNodeParent = pNode->mpNodeParent;
		
		if(pNode == pNodeRoot)
			pNodeRoot = pNodeTemp;
		else if(pNode == pNode->mpNodeParent->mpNodeLeft)
			pNode->mpNodeParent->mpNodeLeft = pNodeTemp;
		else
			pNode->mpNodeParent->mpNodeRight = pNodeTemp;

		pNodeTemp->mpNodeLeft = pNode;
		pNode->mpNodeParent = pNodeTemp;

		return pNodeRoot;
	}



	/// RBTreeRotateRight
	/// Does a right rotation about the given node. 
	/// If you want to understand tree rotation, any book on algorithms will
	/// discussion the topic in good detail.
	template <typename Allocator>
	rbtree_node_base<Allocator>* RBTreeRotateRight(rbtree_node_base<Allocator>* pNode, rbtree_node_base<Allocator>* pNodeRoot)
	{
		rbtree_node_base<Allocator>* const pNodeTemp = pNode->mpNodeLeft;

		pNode->mpNodeLeft = pNodeTemp->mpNodeRight;

		if(pNodeTemp->mpNodeRight)
			pNodeTemp->mpNodeRight->mpNodeParent = pNode;
		pNodeTemp->mpNodeParent = pNode->mpNodeParent;

		if(pNode == pNodeRoot)
			pNodeRoot = pNodeTemp;
		else if(pNode == pNode->mpNodeParent->mpNodeRight)
			pNode->mpNodeParent->mpNodeRight = pNodeTemp;
		else
			pNode->mpNodeParent->mpNodeLeft = pNodeTemp;

		pNodeTemp->mpNodeRight = pNode;
		pNode->mpNodeParent = pNodeTemp;

		return pNodeRoot;
	}




	/// RBTreeInsert
	/// Insert a node into the tree and rebalance the tree as a result of the 
	/// disturbance the node introduced.
	///
	template <typename Allocator>
	EASTL_API void RBTreeInsert(rbtree_node_base<Allocator>* pNode,
								rbtree_node_base<Allocator>* pNodeParent, 
								rbtree_node_base<Allocator>* pNodeAnchor,
								RBTreeSide insertionSide)
	{
		auto & pNodeRootRef = pNodeAnchor->mpNodeParent;

		// Initialize fields in new node to insert.
		pNode->mpNodeParent = pNodeParent;
		pNode->mpNodeRight  = NULL;
		pNode->mpNodeLeft   = NULL;
		pNode->mColor       = RBTreeColor::kRed;

		// Insert the node.
		if(insertionSide == kRBTreeSideLeft)
		{
			pNodeParent->mpNodeLeft = pNode; // Also makes (leftmost = pNode) when (pNodeParent == pNodeAnchor)

			if(pNodeParent == pNodeAnchor)
			{
				pNodeAnchor->mpNodeParent = pNode;
				pNodeAnchor->mpNodeRight = pNode;
			}
			else if(pNodeParent == pNodeAnchor->mpNodeLeft)
				pNodeAnchor->mpNodeLeft = pNode; // Maintain leftmost pointing to min node
		}
		else
		{
			pNodeParent->mpNodeRight = pNode;

			if(pNodeParent == pNodeAnchor->mpNodeRight)
				pNodeAnchor->mpNodeRight = pNode; // Maintain rightmost pointing to max node
		}

		// Rebalance the tree.
		while((pNode != pNodeRootRef) && (pNode->mpNodeParent->mColor == RBTreeColor::kRed))
		{
			EA_ANALYSIS_ASSUME(pNode->mpNodeParent != NULL);
			rbtree_node_base<Allocator>* const pNodeParentParent = pNode->mpNodeParent->mpNodeParent;

			if(pNode->mpNodeParent == pNodeParentParent->mpNodeLeft)
			{
				rbtree_node_base<Allocator>* const pNodeTemp = pNodeParentParent->mpNodeRight;

				if(pNodeTemp && (pNodeTemp->mColor == RBTreeColor::kRed))
				{
					pNode->mpNodeParent->mColor = RBTreeColor::kBlack;
					pNodeTemp->mColor = RBTreeColor::kBlack;
					pNodeParentParent->mColor = RBTreeColor::kRed;
					pNode = pNodeParentParent;
				}
				else 
				{
					if(pNode->mpNodeParent && pNode == pNode->mpNodeParent->mpNodeRight) 
					{
						pNode = pNode->mpNodeParent;
						pNodeRootRef = RBTreeRotateLeft(pNode, pNodeRootRef.value());
					}

					EA_ANALYSIS_ASSUME(pNode->mpNodeParent != NULL);
					pNode->mpNodeParent->mColor = RBTreeColor::kBlack;
					pNodeParentParent->mColor = RBTreeColor::kRed;
					pNodeRootRef = RBTreeRotateRight(pNodeParentParent, pNodeRootRef.value());
				}
			}
			else 
			{
				rbtree_node_base<Allocator>* const pNodeTemp = pNodeParentParent->mpNodeLeft;

				if(pNodeTemp && (pNodeTemp->mColor == RBTreeColor::kRed))
				{
					pNode->mpNodeParent->mColor = RBTreeColor::kBlack;
					pNodeTemp->mColor = RBTreeColor::kBlack;
					pNodeParentParent->mColor = RBTreeColor::kRed;
					pNode = pNodeParentParent;
				}
				else 
				{
					EA_ANALYSIS_ASSUME(pNode != NULL && pNode->mpNodeParent != NULL);

					if(pNode == pNode->mpNodeParent->mpNodeLeft) 
					{
						pNode = pNode->mpNodeParent;
						pNodeRootRef = RBTreeRotateRight(pNode, pNodeRootRef.value());
					}

					pNode->mpNodeParent->mColor = RBTreeColor::kBlack;
					pNodeParentParent->mColor = RBTreeColor::kRed;
					pNodeRootRef = RBTreeRotateLeft(pNodeParentParent, pNodeRootRef.value());
				}
			}
		}

		EA_ANALYSIS_ASSUME(pNodeRootRef != NULL);
		pNodeRootRef->mColor = RBTreeColor::kBlack;

	} // RBTreeInsert




	/// RBTreeErase
	/// Erase a node from the tree.
	///
	template <typename Allocator>
	void RBTreeErase(rbtree_node_base<Allocator>* pNode, rbtree_node_base<Allocator>* pNodeAnchor)
	{
		auto & pNodeRootRef      = pNodeAnchor->mpNodeParent;
		auto & pNodeLeftmostRef  = pNodeAnchor->mpNodeLeft;
		auto & pNodeRightmostRef = pNodeAnchor->mpNodeRight;
		rbtree_node_base<Allocator>*  pNodeSuccessor    = pNode;
		rbtree_node_base<Allocator>*  pNodeChild        = NULL;
		rbtree_node_base<Allocator>*  pNodeChildParent  = NULL;

		if(pNodeSuccessor->mpNodeLeft == nullptr)         // pNode has at most one non-NULL child.
			pNodeChild = pNodeSuccessor->mpNodeRight;  // pNodeChild might be null.
		else if(pNodeSuccessor->mpNodeRight == nullptr)   // pNode has exactly one non-NULL child.
			pNodeChild = pNodeSuccessor->mpNodeLeft;   // pNodeChild is not null.
		else 
		{
			// pNode has two non-null children. Set pNodeSuccessor to pNode's successor. pNodeChild might be NULL.
			pNodeSuccessor = pNodeSuccessor->mpNodeRight;

			while(pNodeSuccessor->mpNodeLeft)
				pNodeSuccessor = pNodeSuccessor->mpNodeLeft;

			pNodeChild = pNodeSuccessor->mpNodeRight;
		}

		// Here we remove pNode from the tree and fix up the node pointers appropriately around it.
		if(pNodeSuccessor == pNode) // If pNode was a leaf node (had both NULL children)...
		{
			pNodeChildParent = pNodeSuccessor->mpNodeParent;  // Assign pNodeReplacement's parent.

			if(pNodeChild) 
				pNodeChild->mpNodeParent = pNodeSuccessor->mpNodeParent;

			if(pNode == pNodeRootRef) // If the node being deleted is the root node...
				pNodeRootRef = pNodeChild; // Set the new root node to be the pNodeReplacement.
			else 
			{
				if(pNode == pNode->mpNodeParent->mpNodeLeft) // If pNode is a left node...
					pNode->mpNodeParent->mpNodeLeft  = pNodeChild;  // Make pNode's replacement node be on the same side.
				else
					pNode->mpNodeParent->mpNodeRight = pNodeChild;
				// Now pNode is disconnected from the bottom of the tree (recall that in this pathway pNode was determined to be a leaf).
			}

			if(pNode == pNodeLeftmostRef) // If pNode is the tree begin() node...
			{
				// Because pNode is the tree begin(), pNode->mpNodeLeft must be NULL.
				// Here we assign the new begin() (first node).
				if(pNode->mpNodeRight && pNodeChild)
				{
					EASTL_ASSERT(pNodeChild != NULL); // Logically pNodeChild should always be valid.
					pNodeLeftmostRef = RBTreeGetMinChild(pNodeChild); 
				}
				else
					pNodeLeftmostRef = pNode->mpNodeParent; // This  makes (pNodeLeftmostRef == end()) if (pNode == root node)
			}

			if(pNode == pNodeRightmostRef) // If pNode is the tree last (rbegin()) node...
			{
				// Because pNode is the tree rbegin(), pNode->mpNodeRight must be NULL.
				// Here we assign the new rbegin() (last node)
				if(pNode->mpNodeLeft && pNodeChild)
				{
					EASTL_ASSERT(pNodeChild != NULL); // Logically pNodeChild should always be valid.
					pNodeRightmostRef = RBTreeGetMaxChild(pNodeChild);
				}
				else // pNodeChild == pNode->mpNodeLeft
					pNodeRightmostRef = pNode->mpNodeParent; // makes pNodeRightmostRef == &mAnchor if pNode == pNodeRootRef
			}
		}
		else // else (pNodeSuccessor != pNode)
		{
			// Relink pNodeSuccessor in place of pNode. pNodeSuccessor is pNode's successor.
			// We specifically set pNodeSuccessor to be on the right child side of pNode, so fix up the left child side.
			pNode->mpNodeLeft->mpNodeParent = pNodeSuccessor; 
			pNodeSuccessor->mpNodeLeft = pNode->mpNodeLeft;

			if(pNodeSuccessor == pNode->mpNodeRight) // If pNode's successor was at the bottom of the tree... (yes that's effectively what this statement means)
				pNodeChildParent = pNodeSuccessor; // Assign pNodeReplacement's parent.
			else
			{
				pNodeChildParent = pNodeSuccessor->mpNodeParent;

				if(pNodeChild)
					pNodeChild->mpNodeParent = pNodeChildParent;

				pNodeChildParent->mpNodeLeft = pNodeChild;

				pNodeSuccessor->mpNodeRight = pNode->mpNodeRight;
				pNode->mpNodeRight->mpNodeParent = pNodeSuccessor;
			}

			if(pNode == pNodeRootRef)
				pNodeRootRef = pNodeSuccessor;
			else if(pNode == pNode->mpNodeParent->mpNodeLeft)
				pNode->mpNodeParent->mpNodeLeft = pNodeSuccessor;
			else 
				pNode->mpNodeParent->mpNodeRight = pNodeSuccessor;

			// Now pNode is disconnected from the tree.

			pNodeSuccessor->mpNodeParent = pNode->mpNodeParent;
			eastl::swap(pNodeSuccessor->mColor, pNode->mColor);
		}

		// Here we do tree balancing as per the conventional red-black tree algorithm.
		if(pNode->mColor == RBTreeColor::kBlack)
		{ 
			while((pNodeChild != pNodeRootRef) && ((pNodeChild == NULL) || (pNodeChild->mColor == RBTreeColor::kBlack)))
			{
				if(pNodeChild == pNodeChildParent->mpNodeLeft) 
				{
					rbtree_node_base<Allocator>* pNodeTemp = pNodeChildParent->mpNodeRight;

					if(pNodeTemp->mColor == RBTreeColor::kRed)
					{
						pNodeTemp->mColor = RBTreeColor::kBlack;
						pNodeChildParent->mColor = RBTreeColor::kRed;
						pNodeRootRef = RBTreeRotateLeft(pNodeChildParent, pNodeRootRef.value());
						pNodeTemp = pNodeChildParent->mpNodeRight;
					}

					if(((pNodeTemp->mpNodeLeft  == nullptr) || (pNodeTemp->mpNodeLeft->mColor  == RBTreeColor::kBlack)) &&
						((pNodeTemp->mpNodeRight == nullptr) || (pNodeTemp->mpNodeRight->mColor == RBTreeColor::kBlack)))
					{
						pNodeTemp->mColor = RBTreeColor::kRed;
						pNodeChild = pNodeChildParent;
						pNodeChildParent = pNodeChildParent->mpNodeParent;
					} 
					else 
					{
						if((pNodeTemp->mpNodeRight == nullptr) || (pNodeTemp->mpNodeRight->mColor == RBTreeColor::kBlack))
						{
							pNodeTemp->mpNodeLeft->mColor = RBTreeColor::kBlack;
							pNodeTemp->mColor = RBTreeColor::kRed;
							pNodeRootRef = RBTreeRotateRight(pNodeTemp, pNodeRootRef.value());
							pNodeTemp = pNodeChildParent->mpNodeRight;
						}

						pNodeTemp->mColor = pNodeChildParent->mColor;
						pNodeChildParent->mColor = RBTreeColor::kBlack;

						if(pNodeTemp->mpNodeRight) 
							pNodeTemp->mpNodeRight->mColor = RBTreeColor::kBlack;

						pNodeRootRef = RBTreeRotateLeft(pNodeChildParent, pNodeRootRef.value());
						break;
					}
				} 
				else 
				{   
					// The following is the same as above, with mpNodeRight <-> mpNodeLeft.
					rbtree_node_base<Allocator>* pNodeTemp = pNodeChildParent->mpNodeLeft;

					if(pNodeTemp->mColor == RBTreeColor::kRed)
					{
						pNodeTemp->mColor        = RBTreeColor::kBlack;
						pNodeChildParent->mColor = RBTreeColor::kRed;

						pNodeRootRef = RBTreeRotateRight(pNodeChildParent, pNodeRootRef.value());
						pNodeTemp = pNodeChildParent->mpNodeLeft;
					}

					if(((pNodeTemp->mpNodeRight == nullptr) || (pNodeTemp->mpNodeRight->mColor == RBTreeColor::kBlack)) &&
						((pNodeTemp->mpNodeLeft  == nullptr) || (pNodeTemp->mpNodeLeft->mColor  == RBTreeColor::kBlack)))
					{
						pNodeTemp->mColor = RBTreeColor::kRed;
						pNodeChild       = pNodeChildParent;
						pNodeChildParent = pNodeChildParent->mpNodeParent;
					} 
					else 
					{
						if((pNodeTemp->mpNodeLeft == nullptr) || (pNodeTemp->mpNodeLeft->mColor == RBTreeColor::kBlack))
						{
							pNodeTemp->mpNodeRight->mColor = RBTreeColor::kBlack;
							pNodeTemp->mColor              = RBTreeColor::kRed;

							pNodeRootRef = RBTreeRotateLeft(pNodeTemp, pNodeRootRef.value());
							pNodeTemp = pNodeChildParent->mpNodeLeft;
						}

						pNodeTemp->mColor = pNodeChildParent->mColor;
						pNodeChildParent->mColor = RBTreeColor::kBlack;

						if(pNodeTemp->mpNodeLeft) 
							pNodeTemp->mpNodeLeft->mColor = RBTreeColor::kBlack;

						pNodeRootRef = RBTreeRotateRight(pNodeChildParent, pNodeRootRef.value());
						break;
					}
				}
			}

			if(pNodeChild)
				pNodeChild->mColor = RBTreeColor::kBlack;
		}

	} // RBTreeErase



} // namespace eastl
























