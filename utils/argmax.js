export function argmax(array){
    return [].reduce.call(array, (_m, _c, _i, _arr) => _c > _arr[_m] ? _i : _m, 0);
}
