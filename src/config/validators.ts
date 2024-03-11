
export class Validators {

    constructor(
        private readonly data: {[key: string]: any}
    ) {}

    public isEmail(key: string) {
        const regular = /^(([^<>()\[\]\\.,;:\s@”]+(\.[^<>()\[\]\\.,;:\s@”]+)*)|(“.+”))@((\[[0–9]{1,3}\.[0–9]{1,3}\.[0–9]{1,3}\.[0–9]{1,3}])|(([a-zA-Z\-0–9]+\.)+[a-zA-Z]{2,}))$/;
        if(!regular.test(this.data[key])) throw 'Correo no valido';
    }

    public requiredKeys( ...keys: string[]) {
        keys.forEach(k => {
            if(!this.data[k]) throw `${k} faltante`;
        })
    }

    public isRequired(key: string) {
        if(!this.data[key]) throw `${key} faltante`;
    }

    public isNumber(key: string) {
        this.isRequired(key);
        if(isNaN(this.data[key])) throw `${key} no es un numero valido`;
        this.data[key] = parseInt(this.data[key]);
    }

    public isFloat(key: string) {
        this.isRequired(key);
        if(isNaN(this.data[key])) throw `${key} no es un numero valido`;
        this.data[key] = parseFloat(this.data[key]);
    }

    public capitalizar(key: string) {
        this.isRequired(key);
        const str = this.data[key] as string;
        const array = str.split(' ');
        array.forEach((s, i) => {
            if(s.length === 0) return;
            s = s.toLowerCase();
            const primarCaracter = s.at(0)?.toUpperCase();
            const restoCadena = s.slice(1);
            array[i] = primarCaracter + restoCadena;
        });
        this.data[key] = array.join(' ');
    }

    public isBoolean(key: string) {
        if( typeof this.data[key] !== 'boolean') throw `${key} no es un boolean valido`;        
    }

    public toUpperCase(key: string) {
        this.isRequired(key);
        this.data[key] = (this.data[key] as string).toUpperCase();
    }

    public isDate(key: string) {
        this.isRequired(key);
        const newDate = new Date(this.data[key]);
        if(newDate.toString() === 'Invalid Date') throw `${key} no es una fecha valida`;
        this.data[key] = newDate;
    }

    public checkPattern(key: string, pattern: RegExp) {
        this.isRequired(key);
        if(!pattern.test(this.data[key])) throw `${key} no valido`;
    }
}