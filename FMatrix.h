#pragma once

template <int r, int c>
class FMatrix final
{
public:
	FMatrix();
	~FMatrix();
	FMatrix(const FMatrix&) = delete;
	FMatrix& operator=(const FMatrix&) = delete;
	FMatrix(FMatrix&&) noexcept = delete;
	FMatrix& operator=(FMatrix&&) noexcept = delete;

private:

};