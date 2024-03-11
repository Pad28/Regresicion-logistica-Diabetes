import { Router } from "express";
import { PredictRoutes } from "./predict/routes";

export class AppRoutes {
    static get routes(): Router {
        const router = Router();

        router.use('/api/predict', PredictRoutes.routes);

        return router;
    }
}