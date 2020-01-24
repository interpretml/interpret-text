export enum RadioKeys {
  all = 'all',
  pos = 'pos',
  neg = 'neg'
}
export class Utils {
  public static argsort(toSort: number[]): number[] {
    const sorted = toSort.map((val, index) => [val, index])
    sorted.sort((a, b) => {
      return a[0] - b[0]
    })
    return sorted.map(val => val[1])
  }

  public static sortedTopK(list:number[], k:number, radio:string): number[] {
    let sortedList: number[]
    if (radio === RadioKeys.all) {
      sortedList = this.takeTopK(this.argsort(list.map(Math.abs)).reverse(), k)
    } else if (radio === RadioKeys.neg) {
      sortedList = this.takeTopK(this.argsort(list), k)
    } else if (radio === RadioKeys.pos) {
      sortedList = this.takeTopK(this.argsort(list).reverse(), k)
    }
    return sortedList
  }

  public static takeTopK(list:number[], k:number){
    return list.splice(0,k).reverse()

  }
}
