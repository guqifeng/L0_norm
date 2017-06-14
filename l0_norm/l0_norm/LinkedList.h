/*
Author:		Rang M. H. Nguyen
Reference:	Rang M. H. Nguyen, Michael S. Brown
			Fast and Effective L0 Gradient Minimization by Region Fusion
			ICCV 2015
Date:		Dec 1st, 2015
*/
//-------------------------------

#pragma once
#include <iostream>
struct Node
{
	int value;
	Node* next;
};

class LinkedList
{
public:
	Node* pHead;
	Node* pTail;
	void append(LinkedList&);
	void insert(int v);
	LinkedList(void);
	void clear();
	~LinkedList(void);
};

