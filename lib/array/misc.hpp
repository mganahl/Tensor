  template<typename T>
  Complex caster(Complex a, T b){
    return Complex(b);
  }

  
  Real caster(Real a, Complex b){
    Warning("casting Complex to Real discards the imaginary part of scalar")
    return b.real();
  }
  Real caster(Real a, Real b){
    return b;
  }
  Real caster(Real a, float b){
    return (Real) b;
  }
  Real caster(Real a, int b){
    return (Real) b;
  }
  Real caster(Real a, lint b){
    return (Real) b;
  }
  Real caster(Real a, luint b){
    return (Real) b;
  }
  Real caster(Real a, uint b){
    return (Real) b;
  }

  
  float caster(float a, Complex b){
    Warning("casting Complex to float discards the imaginary part of scalar")
      return (float)b.real();
  }
  float caster(float a, Real b){
    return (float)b;
  }
  float caster(float a, float b){
    return b;
  }
  float caster(float a, int b){
    return (float) b;
  }
  float caster(float a, lint b){
    return (float) b;
  }
  float caster(float a, luint b){
    return (float) b;
  }
  float caster(float a, uint b){
    return (float) b;
  }

  
  lint caster(lint a, Complex b){
    Warning("casting Complex to lint discards the imaginary part of scalar")
      return (lint)b.real();
  }
  lint caster(lint a, Real b){
    return (lint)b;
  }
  lint caster(lint a, float b){
    return (lint) b;
  }
  lint caster(lint a, int b){
    return (lint) b;
  }
  lint caster(lint a, lint b){
    return b;
  }
  lint caster(lint a, luint b){
    return (lint) b;
  }
  lint caster(lint a, uint b){
    return (lint) b;
  }

  
  uint caster(uint a, Complex b){
    Warning("casting Complex to lint discards the imaginary part of scalar")
      return (unt)b.real();
  }
  uint caster(uint a, Real b){
    return (uint)b;
  }
  uint caster(uint a, float b){
    return (uint) b;
  }
  uint caster(uint a, int b){
    return (uint) b;
  }
  uint caster(uint a, lint b){
    return (uint)b;
  }
  uint caster(uint a, luint b){
    return (uint) b;
  }
  uint caster(uint a, uint b){
    return (uint) b;
  }
  
  int caster(lint a, Complex b){
    Warning("casting Complex to lint discards the imaginary part of scalar")
      return (int)b.real();
  }
  int caster(int a, Real b){
    return (lint)b;
  }
  int caster(int a, float b){
    return (int) b;
  }
  int caster(int a, int b){
    return (int) b;
  }
  int caster(int a, lint b){
    return (int)b;
  }
  int caster(int a, luint b){
    return (int) b;
  }
  int caster(int a, uint b){
    return (int) b;
  }
  
