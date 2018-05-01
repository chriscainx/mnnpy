int comp(const void* a, const void* b)
{
    float fa = **(const float **) a;
    float fb = **(const float **) b;
    return (fa > fb) - (fa < fb);
}
