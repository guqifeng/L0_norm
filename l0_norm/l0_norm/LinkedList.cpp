/*
Author:		Rang M. H. Nguyen
Reference:	Rang M. H. Nguyen, Michael S. Brown
			Fast and Effective L0 Gradient Minimization by Region Fusion
			ICCV 2015
Date:		Dec 1st, 2015
*/
//-------------------------------

#include "LinkedList.h"


LinkedList::LinkedList(void)
{
	pHead = pTail = NULL;
}


LinkedList::~LinkedList(void)
{
	Node* ptr = pHead;
	while(pHead != NULL)
	{
		pHead = pHead->next;
		delete ptr;
		ptr = pHead;
	}
	pTail = NULL;
}

void LinkedList::clear()
{
	pHead = pTail = NULL;
}

void LinkedList::append(LinkedList& l)
{
	pTail->next = l.pHead;
	pTail = l.pTail;
	l.clear();
}

void LinkedList::insert(int v)
{
	Node* node = new Node();
	node->value = v;
	if(pHead == NULL)
	{
		pHead = pTail = node;
	}
	else
	{
		pTail->next = node;
		pTail = node;
	}
}
