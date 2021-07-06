#pragma once
/*
 * helpers.h
 *
 *  Created on: 8 May 2020
 *      Author: jkiesele
 */

#include <iostream>

#define I2D(i,j,Nj) j + Nj*i
#define I3D(i,j,k,Nj,Nk) k + Nk*(j + Nj*i)
#define I4D(i,j,k,l,Nj,Nk,Nl) l + Nl*(k + Nk*(j + Nj*i))
#define I5D(i,j,k,l,m,Nj,Nk,Nl,Nm) m + Nm*(l + Nl*(k + Nk*(j + Nj*i)))

#define DEBUGCOUT(x) {std::cout << #x <<": " << x << std::endl;}
