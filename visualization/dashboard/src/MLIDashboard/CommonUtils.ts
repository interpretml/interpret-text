export class Utils{
    public static argsort(toSort: number[]): number[] {
        const sorted = toSort.map((val, index) => [val, index]);
        sorted.sort((a, b) => {
            return a[0] - b[0];
        });
        return sorted.map(val => val[1]);
        }
}